"""
================================================================================
TRUE 2-HOP PHASE 1: P_c^2 Layer Extraction
Appendix Section 1 — Data Engineering & The 2-Hop Expansion
================================================================================
This script fetches the P_c^2 layer (Hop-2 citations) from the Semantic Scholar
API and saves the results to Parquet files with strict disk-based checkpointing.

Run directly:
    python fetch_hop2.py

Or from a Jupyter notebook cell:
    %run fetch_hop2.py
================================================================================
"""

import pandas as pd
import requests
import json
import os
import time
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PARQUET_ENGINE    = "fastparquet"      
INPUT_PARQUET     = "scicite_training_data.parquet"
OUTPUT_DIR        = "hop2_checkpoints" 
CONSOLIDATED_PATH = "hop2_edges.parquet"  

# Important: Requesting individual citation contexts MUST be done via the single-item
# endpoint because the batch endpoint (/paper/batch) throws HTTP 400 when asked for contexts.
# The API provides citations with pagination. We fetch up to 100 per paper.
FLUSH_EVERY       = 200   # Write a chunk every N *papers* processed
INITIAL_BACKOFF_S = 60    # Seconds to wait after first 429
MAX_BACKOFF_S     = 300   # Exponential back-off cap
MAX_RETRIES       = 5     

S2_API_KEY = None  

# We fetch title, year, abstract, authors, and contexts (citation sentence)
FIELDS = "paperId,title,year,abstract,authors,contexts"
# Using the individual citation endpoint
def _get_api_url(paper_id):
    return f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields={FIELDS}&limit=100"

# ==============================================================================
# HELPERS
# ==============================================================================

def _build_headers() -> dict:
    h = {"Accept": "application/json"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def _flush_to_parquet(edge_buffer: list, ids_to_checkpoint: list, chunk_counter: dict) -> list:
    if not ids_to_checkpoint:
        return []

    if edge_buffer:
        chunk_n    = chunk_counter["n"]
        chunk_path = os.path.join(OUTPUT_DIR, f"hop2_edges_chunk_{chunk_n:04d}.parquet")

        temp_df = pd.DataFrame(edge_buffer)
        
        # We use pyarrow engine here because Pandas 2.x dynamically backs string 
        # columns with ArrowExtensionArrays. fastparquet crashes trying to zero-copy these.
        try:
            temp_df.to_parquet(chunk_path, engine="pyarrow", index=False)
        except Exception:
            # Fallback for systems that don't have pyarrow installed
            try:
                temp_df.to_parquet(chunk_path, engine="fastparquet", index=False)
            except Exception as e:
                # Absolute fail-safe: convert df to parquet manually or drop types
                temp_df.astype(str).to_parquet(chunk_path, engine="fastparquet", index=False)
        
        assert os.path.exists(chunk_path), "Parquet write appeared to succeed but file not found"

        chunk_counter["n"] += 1
        with open(os.path.join(OUTPUT_DIR, "_chunk_manifest.json"), "w") as f:
            json.dump(chunk_counter, f)

    with open(os.path.join(OUTPUT_DIR, "_processed_hop1_ids.txt"), "a") as f:
        for pid in ids_to_checkpoint:
            f.write(str(pid) + "\n")

    return []  


def _fetch_citations_with_retry(paper_id: str) -> dict | None:
    url = _get_api_url(paper_id)
    headers = _build_headers()
    backoff  = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait = min(backoff, MAX_BACKOFF_S)
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue  

            if response.status_code in (400, 404):
                return None  # Paper not found or malformed ID

            time.sleep(5)

        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            time.sleep(min(backoff, MAX_BACKOFF_S))
            backoff = min(backoff * 2, MAX_BACKOFF_S)

    return None

def _parse_hop2_edges(hop1_id: str, api_response: dict) -> list:
    edges = []
    
    if not api_response or "data" not in api_response:
        return edges

    for citation_edge in api_response["data"]:
        citation = citation_edge.get("citingPaper")
        if not citation:
            continue
            
        hop2_id = citation.get("paperId")
        if not hop2_id:
            continue

        contexts = citation_edge.get("contexts") or []
        ctx      = contexts[0] if contexts else None

        raw_authors = citation.get("authors") or []
        author_ids  = [a.get("authorId") for a in raw_authors if a.get("authorId")]

        edges.append({
            "hop1_paper_id"   : str(hop1_id),
            "hop2_paper_id"   : hop2_id,
            "hop2_title"      : citation.get("title"),
            "hop2_year"       : citation.get("year"),
            "hop2_abstract"   : citation.get("abstract"),  
            "hop2_author_ids" : json.dumps(author_ids),     
            "citation_context": ctx,                         
        })
    return edges


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("=" * 70)
    print("STEP 1: Loading P_c^1 (Hop-1 citing papers) as query targets...")
    df = pd.read_parquet(INPUT_PARQUET, engine=PARQUET_ENGINE)
    hop1_ids = df["citingPaperId"].dropna().unique().tolist()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(OUTPUT_DIR, "_processed_hop1_ids.txt")
    
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_ids = {line.strip() for line in f if line.strip()}

    # ==========================================================================
    # TEST MODE: DENSE SEED FOCUS
    # We pick the Top N most cited 'Seed' (Hop-0) papers to ensure a dense 
    # and complete network cluster for testing.
    # Set TEST_NUM_SEEDS to an integer (e.g., 5) or None for full run.
    # ==========================================================================
    TEST_NUM_SEEDS = 5 

    if TEST_NUM_SEEDS is not None:
        # 1. Identify the TOP cited seeds (frequency count)
        seed_counts = df["citedPaperId"].value_counts()
        target_seeds = seed_counts.head(TEST_NUM_SEEDS).index.tolist()
        
        # 2. Get ALL Hop-1 papers citing these specific seeds
        target_hop1_df = df[df["citedPaperId"].isin(target_seeds)]
        target_hop1_ids = target_hop1_df["citingPaperId"].dropna().unique().tolist()
        
        # 3. Filter to only those not yet processed
        remaining = [pid for pid in target_hop1_ids if pid not in processed_ids]
        
        print(f"\n🎯 DENSE SEED MODE: Processing Top {len(target_seeds)} Seed papers.")
        print(f"   These seeds have a total of {len(target_hop1_ids)} Hop-1 citations.")
        print(f"   Remaining to fetch: {len(remaining)}")
    else:
        remaining = [pid for pid in hop1_ids if pid not in processed_ids]
        print(f"\n🚀 FULL MODE: Processing {len(remaining)} papers.")
    
    chunk_counter = {"n": 0}
    manifest_file = os.path.join(OUTPUT_DIR, "_chunk_manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            chunk_counter = json.load(f)

    print(f"Remaining to fetch: {len(remaining):,} / {len(hop1_ids):,}")

    if remaining:
        print("\nSTEP 2: Fetching P_c^2 Layer (Individual Endpoint)...")
        in_memory_edges = []
        ids_since_flush = []
        
        # We process one by one because batch endpoint doesn't support 'contexts'
        for paper_id in tqdm(remaining, desc="Fetching Citations"):
            time.sleep(0.35) # Hardcoded polite delay to help prevent aggressive 429s
            
            api_response = _fetch_citations_with_retry(paper_id)
            if api_response is not None:
                new_edges = _parse_hop2_edges(paper_id, api_response)
                in_memory_edges.extend(new_edges)
            
            ids_since_flush.append(paper_id)
            
            if len(ids_since_flush) >= FLUSH_EVERY:
                in_memory_edges = _flush_to_parquet(in_memory_edges, ids_since_flush, chunk_counter)
                ids_since_flush = []

        if in_memory_edges or ids_since_flush:
            _flush_to_parquet(in_memory_edges, ids_since_flush, chunk_counter)
            
    # Consolidation
    print("\nSTEP 3: Consolidating Parquet chunks...")
    chunk_files = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("hop2_edges_chunk_")])
    
    if chunk_files:
        dfs = [pd.read_parquet(f, engine=PARQUET_ENGINE) for f in chunk_files]
        hop2_df = pd.concat(dfs, ignore_index=True)\
                    .drop_duplicates(subset=["hop1_paper_id", "hop2_paper_id"])\
                    .reset_index(drop=True)
        try:
            hop2_df.to_parquet(CONSOLIDATED_PATH, engine="pyarrow", index=False)
        except Exception:
            try:
                hop2_df.to_parquet(CONSOLIDATED_PATH, engine="fastparquet", index=False)
            except Exception:
                hop2_df.astype(str).to_parquet(CONSOLIDATED_PATH, engine="fastparquet", index=False)
        print(f"✅ Saved consolidated graph data: {len(hop2_df):,} edges.")
    elif os.path.exists(CONSOLIDATED_PATH):
        print("✅ Consolidated file already exists.")

if __name__ == "__main__":
    main()
