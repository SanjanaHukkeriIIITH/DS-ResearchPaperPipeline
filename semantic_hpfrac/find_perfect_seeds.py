import pandas as pd
import requests
import json
from tqdm import tqdm
import os

def find_perfect_seeds():
    print("Loading Scicite data...")
    df = pd.read_parquet('scicite_training_data.parquet', engine='pyarrow')
    
    # 1. Group by Hop-0 and count Hop-1 citations
    seed_counts = df["citedPaperId"].value_counts()
    
    # Pick candidate clusters that have a medium density (e.g., 10 to 30 citations) 
    # to avoid querying 5,000 papers for a massive cluster.
    candidates = seed_counts[(seed_counts >= 10) & (seed_counts <= 30)].head(100).index.tolist()
    print(f"Evaluating {len(candidates)} candidate seed clusters...")
    
    clusters = {}
    all_needed_ids = set()
    
    for seed in candidates:
        hop1_ids = df[df["citedPaperId"] == seed]["citingPaperId"].dropna().unique().tolist()
        clusters[seed] = hop1_ids
        all_needed_ids.add(seed)
        all_needed_ids.update(hop1_ids)
        
    all_needed_ids = list(all_needed_ids)
    print(f"Total unique papers to check on Semantic Scholar: {len(all_needed_ids)}")
    
    # 2. Fetch Abstracts from S2
    metadata = {}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,title,abstract"
    
    for i in tqdm(range(0, len(all_needed_ids), 100), desc="Checking Abstracts"):
        batch = all_needed_ids[i:i+100]
        try:
            res = requests.post(url, json={"ids": batch})
            if res.status_code == 200:
                for paper in res.json():
                    if paper and paper.get("paperId"):
                        # Mark True if abstract exists and is > 20 characters
                        has_abstract = bool(paper.get("abstract") and len(paper.get("abstract")) > 20)
                        has_title = bool(paper.get("title"))
                        metadata[paper["paperId"]] = has_abstract and has_title
        except Exception:
            pass

    # 3. Score the Clusters
    scored_clusters = []
    
    for seed, hop1_ids in clusters.items():
        # Check if the seed itself has an abstract
        if not metadata.get(seed, False):
            continue 
            
        # Check how many Hop-1s have abstracts
        hop1_valid = sum(1 for hid in hop1_ids if metadata.get(hid, False))
        valid_ratio = hop1_valid / len(hop1_ids)
        
        # We just collect them all and sort them to find the true "best case" available
        scored_clusters.append({
            "seed": seed,
            "hop1_count": len(hop1_ids),
            "perfect_hop1_count": hop1_valid,
            "ratio": valid_ratio
        })
            
    # Sort by ratio (descending) and then by total count
    scored_clusters.sort(key=lambda x: (x["ratio"], x["hop1_count"]), reverse=True)
    
    print("\n" + "="*50)
    print("🎯 THE MOST COMPLETE SEED CLUSTERS AVAILABLE")
    print("="*50)
    
    if not scored_clusters:
        print("Could not find any perfect clusters! We might need to relax the 90% threshold.")
        return
        
    perfect_seeds = [c["seed"] for c in scored_clusters[:5]]
    
    for i, c in enumerate(scored_clusters[:5]):
        print(f"{i+1}. Seed ID: {c['seed']}")
        print(f"   Hop-1 completeness: {c['perfect_hop1_count']} / {c['hop1_count']} ({c['ratio']*100:.1f}% abstract coverage)\n")
        
    print(f"You should update TEST_NUM_SEEDS to these specific IDs in fetch_hop2.py!")
    print(f"target_seeds = {perfect_seeds}")

if __name__ == "__main__":
    find_perfect_seeds()
