import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import os

def calc_h_index_from_floats(scores):
    """Computes the Hirsch function H() over an array of continuous/fractional values."""
    c = sorted(list(scores), reverse=True)
    h = 0.0
    for i, score in enumerate(c):
        if score >= i + 1:
            h = float(i + 1)
        else:
            break
    return h

def main():
    print("📊 STARTING GRAPH METRICS COMPUTATION")
    print("=" * 60)

    # 1. Load Data Dependencies
    print("Loading graph data and training outputs...")
    edges_df = pd.read_parquet("data/data_v2/edge_predictions_scaled.parquet")
    logs_df  = pd.read_csv("data/data_v2/training_logs_scaled.csv")
    
    graph_path = "data/data_v2/scicite_hetero_scaled.pt"
    hetero_graph = torch.load(graph_path, weights_only=False)

    # Get the final learned temporal decay parameter
    lambda_val = logs_df.iloc[-1]['lambda']
    print(f"Extracted learned temporal decay (λ): {lambda_val:.4f}")

    # 2. Step 1: Calculate Actual Semantic Weight
    # Formula: Intent_Multiplier * exp(-|lambda| * delta_t)
    intent_scalars = {"Background": 0.5, "Method": 1.5, "Result": 1.2}
    edges_df['intent_multiplier'] = edges_df['predicted_intent'].map(intent_scalars)
    
    edges_df['temporal_decay_factor'] = np.exp(-abs(lambda_val) * edges_df['delta_t'])
    edges_df['semantic_weight'] = edges_df['intent_multiplier'] * edges_df['temporal_decay_factor']

    # 3. Step 2: Compute Base Semantic Score S(c) and Raw Citations for EVERY paper
    print("Computing 1-Hop Metrics (Base Citations & S(c))...")
    
    # S(c) is the sum of incoming semantic weights
    semantic_sums = edges_df.groupby('cited_paper_idx')['semantic_weight'].sum().to_dict()
    # Raw citation is the count of incoming edges
    raw_counts = edges_df.groupby('cited_paper_idx')['citing_paper_idx'].count().to_dict()

    num_total_papers = hetero_graph['paper'].x.size(0)
    S_c_scores = np.array([semantic_sums.get(i, 0.0) for i in range(num_total_papers)])
    raw_citations = np.array([raw_counts.get(i, 0.0) for i in range(num_total_papers)])

    # 4. Step 3: Compute 2-Hop Paper Indices (Normal h_p and Semantic Sh_p)
    print("Executing Phase 1: 2-Hop Topological Traversal (Sh_p)...")
    
    cited_to_citing = defaultdict(list)
    for _, row in edges_df.iterrows():
        cited_to_citing[int(row['cited_paper_idx'])].append(int(row['citing_paper_idx']))

    Sh_scores = np.zeros(num_total_papers)
    normal_h_scores = np.zeros(num_total_papers)

    for p_idx in range(num_total_papers):
        citing_papers = cited_to_citing.get(p_idx, [])
        if not citing_papers:
            continue
        
        # Semantic h-index (Sh_p): Hirsch function over citing papers' S(c)
        citing_semantic_scores = [S_c_scores[c] for c in citing_papers]
        Sh_scores[p_idx] = calc_h_index_from_floats(citing_semantic_scores)

        # Normal h-index (h_p): Hirsch function over citing papers' raw counts
        citing_raw_scores = [raw_citations[c] for c in citing_papers]
        normal_h_scores[p_idx] = calc_h_index_from_floats(citing_raw_scores)

    # 5. Step 4 & 5: Map to Authors and Compute Fractional Allocations
    print("Executing Phase 2: Author-Level Fractional Aggregation...")
    
    writes_edges = hetero_graph['author', 'writes', 'paper'].edge_index.cpu().numpy()
    authors_idx = writes_edges[0]
    papers_idx  = writes_edges[1]

    ap_df = pd.DataFrame({'Author_Idx': authors_idx, 'Paper_Idx': papers_idx})

    paper_stats = pd.DataFrame({
        'Paper_Idx': np.arange(num_total_papers),
        'Raw_h_p': normal_h_scores,
        'Semantic_Sh_p': Sh_scores
    })

    # Find co-authors a_i per paper to prevent hyperauthorship inflation
    coauthors = ap_df.groupby('Paper_Idx').size().reset_index(name='a_i')
    paper_stats = paper_stats.merge(coauthors, on='Paper_Idx', how='left')
    paper_stats['a_i'] = paper_stats['a_i'].fillna(1).replace(0, 1)

    # Calculate fractional fractions (Score / a_i)
    paper_stats['Fractional_Raw_h_p'] = paper_stats['Raw_h_p'] / paper_stats['a_i']
    paper_stats['Fractional_Sh_p'] = paper_stats['Semantic_Sh_p'] / paper_stats['a_i']

    ap_df = ap_df.merge(paper_stats, on='Paper_Idx', how='left')

    # 6. Step 6: Apply Final Author-Level Hirsch Function
    print("Generating Final Author Leaderboards...")
    author_group = ap_df.groupby('Author_Idx')

    author_results = author_group.agg(
        Total_Papers=('Paper_Idx', 'count')
    ).reset_index()

    # Normal hp-frac = H( [ h_p1 / a_1, h_p2 / a_2 ... ] )
    author_results['Normal_hp_frac'] = author_group['Fractional_Raw_h_p'].apply(calc_h_index_from_floats).reset_index(drop=True)
    
    # Semantic hp-frac = H( [ Sh_p1 / a_1, Sh_p2 / a_2 ... ] )
    author_results['Semantic_hp_frac'] = author_group['Fractional_Sh_p'].apply(calc_h_index_from_floats).reset_index(drop=True)

    author_results['Impact_Shift'] = author_results['Semantic_hp_frac'] - author_results['Normal_hp_frac']
    
    # Filter and sort
    final_leaderboard = author_results[author_results['Normal_hp_frac'] > 0].copy()
    final_leaderboard = final_leaderboard.sort_values(by='Semantic_hp_frac', ascending=False).reset_index(drop=True)

    print("\n🏆 TOP 10 AUTHORS BY TRUE SEMANTIC HP-FRAC")
    print("=" * 60)
    print(final_leaderboard[['Author_Idx', 'Total_Papers', 'Normal_hp_frac', 'Semantic_hp_frac', 'Impact_Shift']].head(10).to_string())

    final_leaderboard.to_csv("data/data_v2/author_rankings.csv", index=False)
    print("\n✅ Successfully exported metrics to data/data_v2/author_rankings.csv")

if __name__ == "__main__":
    main()