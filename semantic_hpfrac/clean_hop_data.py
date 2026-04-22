import pandas as pd
import os

def clean_and_stitch():
    print("🧹 STARTING GENTLE HOP DATA CLEANUP...")
    
    # 1. Load Raw Files
    # Check for the v2 file prepared by fetch_ghost_nodes.py first
    hop0_file = "data/hop0_metadata_final_v2.csv" if os.path.exists("data/hop0_metadata_final_v2.csv") else "data/hop0_metadata_final.csv"
    print(f"Loading Hop-0 from: {hop0_file}")
    
    h0 = pd.read_csv(hop0_file)
    h1 = pd.read_csv("data/hop1_final_dataset_rescued.csv")
    h2 = pd.read_csv("data/hop2_final_dataset.csv")

    print(f"Initial counts: Hop0={len(h0)}, Hop1={len(h1)}, Hop2={len(h2)}")

    # 2. Basic Formatting: Strip invisible whitespaces from IDs to prevent false mismatches
    h0['source_paper_id'] = h0['source_paper_id'].astype(str).str.strip()
    
    h1['hop1_id'] = h1['hop1_id'].astype(str).str.strip()
    h1['hop0_id'] = h1['hop0_id'].astype(str).str.strip()
    
    h2['hop2_id'] = h2['hop2_id'].astype(str).str.strip()
    h2['hop1_id'] = h2['hop1_id'].astype(str).str.strip()

    # 3. Intra-file Deduplication (Remove exact duplicate rows within the same file)
    h0_clean = h0.drop_duplicates(subset=['source_paper_id']).copy()
    
    # For Hop-1, prioritize the labeled version if there are multiple entries for the same citing-cited pair
    # (By sorting by intent_label first, labels come before 'unknown')
    h1_clean = h1.drop_duplicates(subset=['hop1_id', 'hop0_id']).copy()
    h2_clean = h2.drop_duplicates(subset=['hop2_id', 'hop1_id']).copy()

    print("\n🔍 After within-file deduplication:")
    print(f"  Hop0: {len(h0)} -> {len(h0_clean)}")
    print(f"  Hop1: {len(h1)} -> {len(h1_clean)}")
    print(f"  Hop2: {len(h2)} -> {len(h2_clean)}")

    # 4. Inter-file Deduplication (Cross-layer Edge Matching)
    # Create a standardized "Edge String" (source_target) to compare exact edges across CSVs
    h1_clean['edge_key'] = h1_clean['hop1_id'] + "_" + h1_clean['hop0_id']
    h2_clean['edge_key'] = h2_clean['hop2_id'] + "_" + h2_clean['hop1_id']

    hop1_edges = set(h1_clean['edge_key'])
    
    # If the EXACT same edge exists in both Hop-1 and Hop-2, we drop it from Hop-2 (prefer prior layer)
    h2_final = h2_clean[~h2_clean['edge_key'].isin(hop1_edges)].copy()
    
    cross_layer_dupes = len(h2_clean) - len(h2_final)
    print(f"🔍 Dropped {cross_layer_dupes} Hop-2 edges because they already exist identically in Hop-1.")

    # Clean up temporary edge_key columns
    h1_final = h1_clean.drop(columns=['edge_key'])
    h2_final = h2_final.drop(columns=['edge_key'])

    # 5. Export Clean Versions
    os.makedirs("data/cleaned", exist_ok=True)
    h0_clean.to_csv("data/cleaned/hop0_v2.csv", index=False)
    h1_final.to_csv("data/cleaned/hop1_v2.csv", index=False)
    h2_final.to_csv("data/cleaned/hop2_v2.csv", index=False)

    print("\n✅ GENTLE CLEANUP COMPLETE!")
    print(f"   Hop0 (seeds):     {len(h0_clean):>7}")
    print(f"   Hop1 (layer 1):   {len(h1_final):>7}")
    print(f"   Hop2 (layer 2):   {len(h2_final):>7}")
    print("\nFiles safely saved to: data/cleaned/hop*_v2.csv")

if __name__ == "__main__":
    clean_and_stitch()
