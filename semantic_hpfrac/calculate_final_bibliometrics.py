import pandas as pd
import os

# ==============================================================================
# Final Bibliometrics (FIXED: Local-Join Version)
# ==============================================================================
print("\n🚀 Calculating Final Bibliometrics from Local Context...")

# Paths to your truth data
impact_path = "data/paper_impact_scaled.csv"
meta_path   = "data/hop0_metadata_final_v2.csv"

if not os.path.exists(impact_path):
    print(f"❌ Error: {impact_path} not found. Please run 'train_hgnn_scaled.py' first.")
    exit()

if not os.path.exists(meta_path):
    print(f"❌ Error: {meta_path} not found. Please run 'fetch_ghost_nodes.py' first.")
    exit()

# 1. Load the Model-Generated Impact
impact_df = pd.read_csv(impact_path)

# 2. Load the Rescued Metadata
meta_df = pd.read_csv(meta_path)

print(f"✅ Loaded {len(impact_df):,} paper scores.")
print(f"✅ Loaded {len(meta_df):,} paper metadata records.")

# 3. Join Model Scores with Metadata
# We merge on the original IDs to ensure S2 Aliases don't break the mapping
final_results = pd.merge(
    impact_df, 
    meta_df[['paper_id', 'title', 'authors', 'year']], 
    left_on='cited_paper_id', 
    right_on='paper_id', 
    how='inner'
)

if final_results.empty:
    print("❌ MAPPING ERROR: Simple join failed. Attempting fuzzy match on original source IDs...")
    # Fallback to source_paper_id check (useful for rescued ghost nodes)
    final_results = pd.merge(
        impact_df,
        meta_df[['source_paper_id', 'title', 'authors', 'year']],
        left_on='cited_paper_id',
        right_on='source_paper_id',
        how='inner'
    ).rename(columns={'source_paper_id': 'paper_id'})

if not final_results.empty:
    # 4. Calculate Author-Level Aggregates
    print("📊 Generating Author Leaderboard...")
    
    # Process authors (handle comma-separated strings)
    auth_impact = final_results.copy()
    auth_impact['author_list'] = auth_impact['authors'].astype(str).str.split(',')
    auth_impact = auth_impact.explode('author_list')
    auth_impact['author_list'] = auth_impact['author_list'].str.strip()
    
    # Group by author and sum impact
    author_leaderboard = auth_impact.groupby('author_list').agg({
        'total_semantic_impact': 'sum',
        'raw_citation_count': 'sum',
        'paper_id': 'count'
    }).rename(columns={'paper_id': 'local_paper_count'}).sort_values('total_semantic_impact', ascending=False)
    
    # Remove any generic empty author names
    author_leaderboard = author_leaderboard[author_leaderboard.index.str.len() > 1]

    print("\n🏆 TOP 10 AUTHORS BY SEMANTIC IMPACT (140k Graph):")
    print(author_leaderboard.head(10))
    
    # 5. Save Final Export
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    author_leaderboard.to_csv(f"{output_dir}/final_author_leaderboard.csv")
    final_results.to_csv(f"{output_dir}/final_citation_analytics.csv", index=False)
    
    print(f"\n✅ Success! Reports saved to '{output_dir}/' directory.")
    print("   - final_author_leaderboard.csv")
    print("   - final_citation_analytics.csv")
else:
    print("❌ Critical Error: Could not link model results to any known paper metadata.")
    print("   Check if 'cited_paper_id' in your model output matches 'paper_id' in your V2 CSV.")
