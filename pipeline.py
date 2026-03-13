import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, lit, concat_ws, sha2, expr, explode, sum as spark_sum, split, length, collect_list, size
from pyspark.ml.feature import HashingTF, IDF

import sys

def setup_hadoop_env():
    """Checks and warns about Hadoop configuration on Windows."""
    if os.name == 'nt':  # Windows
        hadoop_home = os.environ.get("HADOOP_HOME")
        if not hadoop_home:
            print("WARNING: HADOOP_HOME environment variable is not set.")
            print("Spark requires Hadoop binaries (winutils.exe) to run on Windows.")
            print("Please download them from https://github.com/cdarlint/winutils and set HADOOP_HOME.")
        elif not os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            print(f"WARNING: winutils.exe not found in {os.path.join(hadoop_home, 'bin')}")
            print("Spark may fail to perform local IO operations.")

def create_spark_session():
    return SparkSession.builder \
        .appName("Scholarly Data Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def process_arxiv(spark, file_path):
    df = spark.read.json(file_path)
    
    # Extract year from update_date string e.g. '2008-11-26'
    df = df.withColumn("year", expr("cast(substring(update_date, 1, 4) as int)"))
    
    # Process authors_parsed array of arrays
    df = df.withColumn("author_str", expr("transform(authors_parsed, x -> trim(concat_ws(' ', x)))"))
    
    res = df.select(
        col("title"),
        col("abstract"),
        col("author_str").alias("authors"),
        col("year")
    ).withColumn("source", lit("arxiv"))
    
    return res

def process_s2orc(spark, file_path):
    df = spark.read.json(file_path)
    
    # authors is Array of Structs [{"first": "Alice", "last": "Smith"}]
    df = df.withColumn("author_str", expr("transform(authors, x -> trim(concat_ws(' ', x.first, x.last)))"))
    
    res = df.select(
        col("title"),
        col("abstract"),
        col("author_str").alias("authors"),
        col("year")
    ).withColumn("source", lit("s2orc"))
    
    return res

def main():
    setup_hadoop_env()
    spark = create_spark_session()
    
    print("Loading datasets...")
    arxiv_file = "live_arxiv.jsonl" if os.path.exists("live_arxiv.jsonl") else "sample_arxiv.json"
    s2orc_file = "live_s2orc.jsonl" if os.path.exists("live_s2orc.jsonl") else "sample_s2orc.json"
    
    arxiv_df = process_arxiv(spark, arxiv_file)
    s2orc_df = process_s2orc(spark, s2orc_file)
    
    print("Combining datasets...")
    combined_df = arxiv_df.unionByName(s2orc_df)
    
    print("\n--- Day 6: Data Quality Layer ---")
    print("Step 1: Running Validations")
    raw_df = combined_df
    
    valid_df = raw_df.filter(
        (col("abstract").isNotNull()) &
        (col("year").isNotNull()) &
        (size(col("authors")) > 0)
    )
    
    print("Step 2 & 3: Capturing Rejected Records with Reason")
    rejected_df = raw_df.subtract(valid_df)
    rejected_df = rejected_df.withColumn("reason", lit("failed_validation"))
            
    print("Step 4: Persisting Rejection Index")
    rejected_df.write \
        .mode("overwrite") \
        .parquet("index/rejected_papers")
        
    print("Step 5: Continuing Normalization with Valid Records")
    combined_df = valid_df
    
    # Lowercase title and abstract
    combined_df = combined_df.withColumn("title", lower(col("title"))) \
                             .withColumn("abstract", lower(col("abstract")))
                             
    # Paper_id generated via hash(title + year)
    combined_df = combined_df.withColumn("paper_id", sha2(concat_ws("", col("title"), col("year")), 256))
    
    # Deduplicate universally based on the hashed paper ID!
    print("Zero-Redundancy Deduplication...")
    combined_df = combined_df.dropDuplicates(["paper_id"])
    
    # Final unified schema layout
    final_df = combined_df.select("paper_id", "title", "abstract", "authors", "year", "source")
    
    print("Step 1: Writing data to Logical Index...")
    final_df.write \
        .partitionBy("year") \
        .mode("overwrite") \
        .parquet("index/papers")
    
    print("Step 2: Validating Index...")
    indexed_df = spark.read.parquet("index/papers")
    print("Indexed record count:", indexed_df.count())
    indexed_df.show(5, truncate=False)
    
    print("\n--- Day 3: Author Aggregation Layer ---")
    print("Step 1: Exploding Authors Array")
    authors_df = indexed_df.withColumn("author", explode(col("authors")))
    
    print("Step 2: Grouping and Counting by Author and Year")
    author_stats = authors_df.groupBy("author", "year") \
                             .count() \
                             .withColumnRenamed("count", "paper_count")
                             
    print("Step 3: Persisting Author Index to Parquet")
    author_stats.write \
        .partitionBy("year") \
        .mode("overwrite") \
        .parquet("index/author_stats")
        
    print("Step 4: Validating and running Demo Queries")
    loaded_author_stats = spark.read.parquet("index/author_stats")
    
    print("--> Top Authors Overall:")
    loaded_author_stats.groupBy("author") \
        .agg(spark_sum("paper_count").alias("total_papers")) \
        .orderBy(col("total_papers").desc()) \
        .show()
        
    print("--> Year-over-Year Activity for Alice Smith:")
    loaded_author_stats.filter(col("author") == "Alice Smith") \
        .orderBy("year") \
        .show()
        
    print("\n--- Day 5: Enhanced Temporal Topic Aggregation Layer ---")
    print("Step 1: Tokenizing Abstracts")
    tokens_df = indexed_df.withColumn("token", explode(split(lower(col("abstract")), " ")))
    
    print("Step 2: Cleaning Tokens with Stopwords")
    stopwords = [
        "the","and","for","with","this","that","from","are",
        "was","were","into","their","have","has","had","using",
        "use","used","paper","study","method","results","based",
        "a", "in", "is", "of", "to", "on", "as", "by", "at", "an", "be"
    ]
    tokens_df = tokens_df.filter(
        (~col("token").isin(stopwords)) &
        (length(col("token")) > 3)
    )
    
    print("Step 3: Calculating TF-IDF")
    doc_tokens = tokens_df.groupBy("paper_id", "year") \
        .agg(collect_list("token").alias("tokens"))
        
    hashingTF = HashingTF(inputCol="tokens", outputCol="tf_features", numFeatures=5000)
    tf_df = hashingTF.transform(doc_tokens)
    
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    idf_model = idf.fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)
    
    print("Step 4: Grouping by Year and Token (Raw Counts for Index)")
    temporal_topics = tokens_df.groupBy("year", "token").count()
    
    print("Step 5: Persisting Temporal Index to Parquet")
    temporal_topics.write \
        .partitionBy("year") \
        .mode("overwrite") \
        .parquet("index/temporal_topics")
        
    print("Step 6: Validating Enhanced Temporal Topics")
    topics_df = spark.read.parquet("index/temporal_topics")
    print("--> Top Topics in 2023 (after stopword filtering):")
    topics_df.filter(col("year") == 2023) \
        .orderBy(col("count").desc()) \
        .show(10)
        
    print("--> Trend of Keyword 'quantum':")
    topics_df.filter(col("token") == "quantum") \
        .orderBy("year") \
        .show()
        
    print("\n--- Phase 3: Author Collaboration Graph ---")
    print("Step 1: Filtering Multi-Author Papers")
    multi_author_df = final_df.filter(size(col("authors")) > 1).select("paper_id", "authors")
    
    print("Step 2: Exploding and Self-Joining to find Pairs")
    # Explode once to get (paper_id, author1)
    a1 = multi_author_df.select("paper_id", explode("authors").alias("author1"))
    # Explode again to get (paper_id, author2)
    a2 = multi_author_df.select("paper_id", explode("authors").alias("author2"))
    
    # Inner join on paper_id where author1 < author2 to prevent A-B and B-A duplicates and A-A self loops
    pairs = a1.join(a2, on="paper_id").filter(col("author1") < col("author2"))
    
    print("Step 3: Calculating Collaboration Weights")
    collaborations = pairs.groupBy("author1", "author2").count().withColumnRenamed("count", "weight")
    
    print("Step 4: Persisting Graph Index")
    collaborations.write \
        .mode("overwrite") \
        .parquet("index/collaborations")
        
    print("Step 5: Validating Collaboration Graph")
    collab_df = spark.read.parquet("index/collaborations")
    print("--> Top Collaborations globally:")
    collab_df.orderBy(col("weight").desc()).show(10, truncate=False)
    
    spark.stop()

if __name__ == "__main__":
    main()
