# Scholarly Research Data Pipeline & Explorer 🔬

A modular scholarly data pipeline built with **Apache Spark**, featuring real-time web aggregation, temporal topic modeling (TF-IDF), and interactive author collaboration networks.

## 🚀 Overview
This system implements a production-grade data engineering architecture:
1. **Web Sources Aggregator**: Multi-threaded API scraping (arXiv, Semantic Scholar) with local query caching.
2. **Spark Processing Engine**: Unified schema normalization, data quality validation, and temporal indexing.
3. **Analytics Layers**: 
   - **Author Stats**: Year-over-year publishing velocity.
   - **Temporal Topics**: TF-IDF based keyword trend analysis.
   - **Collaboration Graph**: Cartesian self-join to extract research networks.
4. **Interactive UI**: A Streamlit dashboard to explore indices and trigger live web scrapes.

---

## 🛠 Prerequisites

### 1. Java Runtime (Required for Spark)
Apache Spark requires Java 8, 11, or 17. 
- **macOS (Homebrew)**: `brew install openjdk@17`
- **Linux**: `sudo apt install openjdk-17-jdk`

### 2. Python 3.9+
The system is tested on Python 3.13.

---

## 📦 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <project-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Environment Configuration

Before running Spark-based scripts, ensure the following environment variables are set (example for macOS Homebrew OpenJDK 17):

```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export SPARK_LOCAL_IP="127.0.0.1"
```

---

## 🏃 Usage Guide

### Step 1: Aggregate Data (Optional)
Fetch real research papers from the web based on keywords:
```bash
python aggregator.py "quantum computing" "neural networks"
```

### Step 2: Run the Pipeline
Process raw data into persistent Parquet indexes:
```bash
python pipeline.py
```

### Step 3: Launch the UI
Start the interactive dashboard:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
- `aggregator.py`: Multi-threaded API scraper.
- `pipeline.py`: The core Apache Spark ETL engine.
- `app.py`: Streamlit dashboard.
- `index/`: Directory containing partitioned Parquet analytical indexes.
- `aggregator_cache.json`: Local cache to prevent duplicate API calls.
