import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def setup_hadoop_env():
    """Checks and warns about Hadoop configuration on Windows."""
    if os.name == 'nt':  # Windows
        hadoop_home = os.environ.get("HADOOP_HOME")
        if not hadoop_home:
            st.warning("HADOOP_HOME environment variable is not set. Spark requires Hadoop binaries (winutils.exe) to run on Windows.")
            st.info("Please download them from https://github.com/cdarlint/winutils and set HADOOP_HOME.")
        elif not os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            st.error(f"winutils.exe not found in {os.path.join(hadoop_home, 'bin')}. Spark may fail to perform local IO operations.")

# Optional: suppress warnings for simple logging
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)

# ---------------------------------------------------------
# 1. SPARK INIT & DATA LOADING
# ---------------------------------------------------------
@st.cache_resource # Streamlit caches this so Spark doesn't restart on every click
def init_spark():
    setup_hadoop_env()
    return SparkSession.builder \
        .appName("Pipeline UI") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

@st.cache_resource
def load_data(_spark):
    """Loads all Parquet indexes into memory."""
    try:
        papers_df = _spark.read.parquet("index/papers")
        authors_df = _spark.read.parquet("index/author_stats")
        topics_df = _spark.read.parquet("index/temporal_topics")
        collab_df = _spark.read.parquet("index/collaborations")
        return papers_df, authors_df, topics_df, collab_df
    except Exception as e:
        st.error(f"Error loading indexes: {e}. Did you run pipeline.py?")
        return None, None, None, None

# ---------------------------------------------------------
# 2. UI LAYOUT & STATE
# ---------------------------------------------------------
st.set_page_config(page_title="Scholarly Research Explorer", layout="wide")
st.title("🔬 Scholarly Research Explorer")
st.markdown("Phase 2 UI: Running completely on **PySpark** & **Parquet Indexes**.")

spark = init_spark()
papers_df, authors_df, topics_df, collab_df = load_data(spark)

if not papers_df:
    st.stop()

# ---------------------------------------------------------
# 3. INTERACTIVE MODULES
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["📄 Paper Keyword Search", "👨‍🔬 Author Analytics", "📈 Topic Trends", "🕸️ Collaboration Network"])

# Module 1: Paper Keyword Search
with tab1:
    st.header("Search Papers")
    query = st.text_input("Enter a keyword to search within abstracts (e.g., 'quantum'):")
    
    if query:
        # Search the base papers index
        results = papers_df.filter(col("abstract").contains(query.lower()))
        count = results.count()
        
        st.success(f"Found {count} matching papers.")
        if count > 0:
            # Convert Spark DataFrame to Pandas for beautiful Streamlit table rendering
            pdf = results.select("title", "authors", "year", "source").toPandas()
            
            # Neaten the text for presentation!
            pdf["title"] = pdf["title"].str.title()
            pdf["authors"] = pdf["authors"].apply(lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else x)
            pdf["source"] = pdf["source"].str.upper()
            
            st.dataframe(pdf, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("Not satisfied with the results? Tell the web crawler to fetch new papers directly into our Parquet data engine!")
        if st.button(f"🌐 Web Scrape additional papers for '{query}'"):
            with st.spinner(f"Aggregating live papers for '{query}' and re-indexing Spark database... Please wait."):
                import subprocess
                import os
                
                # Copy the existing environment and inject Spark requirements
                env = os.environ.copy()
                env["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
                env["SPARK_LOCAL_IP"] = "127.0.0.1"
                env["PYTHONUNBUFFERED"] = "1"
                
                # Run the threadpool aggregator
                subprocess.run(["python", "aggregator.py", query], env=env, check=True)
                # Process the data through the Spark Engine
                subprocess.run(["python", "pipeline.py"], env=env, check=True)
                
                # Invalidate the cache to reload PySpark
                st.cache_resource.clear()
                st.rerun()

# Module 2: Author Analytics
with tab2:
    st.header("Author Output Analytics")
    
    # Get a list of all unique authors to populate the dropdown
    unique_authors = authors_df.select("author").distinct().toPandas()["author"].tolist()
    
    selected_author = st.selectbox("Select an Author to audit:", [""] + sorted(unique_authors))
    
    if selected_author:
        # Query total global count
        author_data = authors_df.filter(col("author") == selected_author)
        total_papers = author_data.selectExpr("sum(paper_count)").collect()[0][0]
        st.metric(label="Total Papers Published", value=total_papers)
        
        # Query year-over-year publishing velocity
        st.subheader("Publishing Velocity over Time")
        velocity_df = author_data.select("year", "paper_count").orderBy("year").toPandas()
        st.bar_chart(data=velocity_df, x="year", y="paper_count")

# Module 3: Topic Trends
with tab3:
    st.header("Temporal Topic Trends")
    st.markdown("Analyze how frequently specific concepts appear across the years.")
    
    topic_query = st.text_input("Search a keyword trend (e.g., 'learning' or 'quantum'):")
    
    if topic_query:
        # Query the temporal index directly
        trend_data = topics_df.filter(col("token") == topic_query.lower()).orderBy("year")
        count = trend_data.count()
        
        if count > 0:
            st.success(f"Tracked data points for '{topic_query}'.")
            trend_pdf = trend_data.select("year", "count").toPandas()
            st.line_chart(data=trend_pdf, x="year", y="count")
        else:
            st.warning("Keyword not found in the temporal index.")
            
# Module 4: Collaboration Network
with tab4:
    st.header("Author Collaboration Graph")
    st.markdown("Discover the web of relationships between researchers. (Powered by PyVis & PySpark)")
    
    # Get a list of all unique authors for the dropdown
    all_collab_authors = collab_df.select("author1").union(collab_df.select("author2")).distinct().toPandas()["author1"].tolist()
    
    graph_author = st.selectbox("Select an Author to build their network:", [""] + sorted(all_collab_authors), key="g_auth")
    
    if graph_author:
        # Filter for all edges containing this author
        edges = collab_df.filter((col("author1") == graph_author) | (col("author2") == graph_author)) \
            .orderBy(col("weight").desc()) \
            .limit(50) \
            .toPandas()
            
        if len(edges) > 0:
            st.success(f"Built network for {graph_author} from {len(edges)} strong collaborations.")
            from pyvis.network import Network
            import streamlit.components.v1 as components
            
            # Initialize PyVis network with an elegant, modern theme
            net = Network(height='600px', width='100%', bgcolor='#FFFFFF', font_color='#333333')
            
            # Apply smooth physics to prevent chaotic "rave" bouncing
            net.repulsion(node_distance=150, central_gravity=0.1, spring_length=150)
            
            # Add central node (Main Author) - warm sophisticated color
            net.add_node(graph_author, label=graph_author, color='#E67E22', size=35, 
                         borderWidth=2)
            
            for _, row in edges.iterrows():
                # Determine who the collaborator is relative to the central author
                collab_name = row["author2"] if row["author1"] == graph_author else row["author1"]
                
                # Add node (Collaborator) - cool professional color
                net.add_node(collab_name, label=collab_name, color='#3498DB', size=20, 
                             borderWidth=1.5)
                
                # Add smooth grey edges that highlight when hovered
                net.add_edge(graph_author, collab_name, value=row["weight"], 
                             title=f"Co-authored {row['weight']} papers",
                             color={'color': '#BDC3C7', 'highlight': '#7F8C8D'})
            
            # Save and render
            net.save_graph("collab_graph.html")
            with open("collab_graph.html", 'r', encoding='utf-8') as f:
                source_code = f.read()
            components.html(source_code, height=650)
        else:
            st.info("This author has no strong collaborations recorded in the global index yet.")
            
# Note: Spark session remains open in the background to serve queries instantly.
