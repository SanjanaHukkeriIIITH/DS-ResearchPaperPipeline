from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def create_spark_session():
    return SparkSession.builder \
        .appName("Pipeline Queries") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def main():
    spark = create_spark_session()
    
    # 1. Load the pre-processed Parquet indexes instantly
    try:
        papers_df = spark.read.parquet("index/papers")
        authors_df = spark.read.parquet("index/author_stats")
        topics_df = spark.read.parquet("index/temporal_topics")
        rejected_df = spark.read.parquet("index/rejected_papers")
        print("\n✅ Successfully loaded indexes!")
    except Exception as e:
        print("\n❌ Error loading indexes. Did you run pipeline.py first?")
        return
        
    # --- WRITE YOUR CUSTOM QUERIES BELOW ---
    
    # Example Query 1: Find all papers published in 2023
    print("\n--- Papers published in 2023 ---")
    papers_df.filter(col("year") == 2023).show(truncate=False)
    
    # Example Query 2: Search for a specific word in abstract
    print("\n--- Papers mentioning 'quantum' ---")
    papers_df.filter(col("abstract").contains("quantum")).select("title", "year").show(truncate=False)

    # Example Query 3: Show Authors with exactly 1 paper
    print("\n--- Authors with 1 paper ---")
    authors_df.filter(col("paper_count") == 1).show()
    
    # Example Query 4: Inspect rejected records
    print("\n--- Rejected Records ---")
    print(f"Total rejected: {rejected_df.count()}")
    rejected_df.select("title", "reason", "source").show(truncate=False)
    
    # ---------------------------------------
    
    # Bonus: You can also use raw SQL if you prefer!
    papers_df.createOrReplaceTempView("papers")
    
    print("\n--- Spark SQL Query Result ---")
    spark.sql("""
        SELECT source, COUNT(*) as source_count 
        FROM papers 
        GROUP BY source
    """).show()

    spark.stop()

if __name__ == "__main__":
    main()
