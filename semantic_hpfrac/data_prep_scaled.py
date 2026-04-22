import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
import tqdm

def impute_missing(df, text_col, title_col, year_col, default_year=2020):
    """Safely imputes missing text fields and years."""
    # Impute year
    df[year_col] = df[year_col].fillna(default_year)
    
    # Impute abstract/text with title, or fallback string
    df[text_col] = df[text_col].fillna(df[title_col])
    df[text_col] = df[text_col].fillna("No abstract available")
    df[title_col] = df[title_col].fillna("Unknown Title")
    return df

def main():
    print("Loading datasets...")
    hop0 = pd.read_csv("data/hop0_metadata_final.csv")
    hop1 = pd.read_csv("data/hop1_final_dataset_rescued.csv")
    hop2 = pd.read_csv("data/hop2_final_dataset.csv")

    print("Calculating robust temporal defaults...")
    # Get a real median from the dataset to avoid hardcoded fallbacks
    combined_years = pd.concat([
        hop0['year'], 
        hop1['hop1_year'], 
        hop2['hop2_year']
    ]).dropna()
    
    if not combined_years.empty:
        global_median_year = int(combined_years.median())
    else:
        global_median_year = 2018 # Fallback if everything is empty
    
    print(f"Using Global Median Year for imputation: {global_median_year}")

    print("Imputing missing values...")
    hop0 = impute_missing(hop0, "abstract", "title", "year", default_year=global_median_year)
    hop1 = impute_missing(hop1, "hop1_abstract", "hop1_title", "hop1_year", default_year=global_median_year)
    hop2 = impute_missing(hop2, "hop2_abstract", "hop2_title", "hop2_year", default_year=global_median_year)

    print("Collecting unique papers...")
    papers = {}
    
    # Collect all papers sequentially across the loops
    for _, row in hop0.iterrows():
        pid = row["source_paper_id"]
        papers[pid] = {"title": row["title"], "abstract": row["abstract"], "year": row["year"]}
        
    for _, row in hop1.iterrows():
        pid = row["hop1_id"]
        if pid not in papers:
            papers[pid] = {"title": row["hop1_title"], "abstract": row["hop1_abstract"], "year": row["hop1_year"]}
        hop0_pid = row["hop0_id"]
        if hop0_pid not in papers: 
            # Fallback for papers missing from hop0 metadata
            papers[hop0_pid] = {"title": "Unknown Title", "abstract": "No abstract available", "year": global_median_year}
            
    for _, row in hop2.iterrows():
        pid = row["hop2_id"]
        if pid not in papers:
            papers[pid] = {"title": row["hop2_title"], "abstract": row["hop2_abstract"], "year": row["hop2_year"]}

    paper_ids = list(papers.keys())
    paper2idx = {pid: i for i, pid in enumerate(paper_ids)}
    num_papers = len(paper_ids)
    print(f"Total unique papers: {num_papers}")

    print("Loading SciBERT...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    embeddings = []
    batch_size = 16
    
    texts = [str(papers[pid]["title"]) + " [SEP] " + str(papers[pid]["abstract"]) for pid in paper_ids]
    
    print("Computing embeddings...")
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch_text = texts[i:i+batch_size]
        inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(emb)

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Constructing HeteroData object...")
    data = HeteroData()
    data['paper'].x = embeddings
    data['author'].x = torch.zeros(0, 128)

    # Building Edges
    cites_src, cites_dst, edge_attrs, delta_ts, intent_labels, label_masks = [], [], [], [], [], []
    intent_map = {"background": 0, "method": 1, "result": 2}
    
    # Hop1 -> Hop0 (Cites)
    for _, row in hop1.iterrows():
        src = row["hop1_id"]
        dst = row["hop0_id"]
        if src in paper2idx and dst in paper2idx:
            cites_src.append(paper2idx[src])
            cites_dst.append(paper2idx[dst])
            edge_attrs.append(embeddings[paper2idx[src]])
            
            label_str = str(row["intent_label"]).lower()
            if label_str in intent_map:
                intent_labels.append(intent_map[label_str])
                label_masks.append(True)
            else:
                intent_labels.append(-1)
                label_masks.append(False)
                
            # Improved Time Delta calculation
            src_yr = int(float(papers[src]["year"]))
            dst_yr = int(float(papers[dst]["year"]))
            
            # If the cited paper is 'future' due to imputation, force it to src_yr-1
            if dst_yr >= src_yr:
                dst_yr = src_yr - 1
            
            delta_ts.append(max(1, src_yr - dst_yr))

    # Hop2 -> Hop1 (Cites)
    for _, row in hop2.iterrows():
        src = row["hop2_id"]
        dst = row["hop1_id"]
        if src in paper2idx and dst in paper2idx:
            cites_src.append(paper2idx[src])
            cites_dst.append(paper2idx[dst])
            edge_attrs.append(embeddings[paper2idx[src]])
            intent_labels.append(-1)
            label_masks.append(False)
            
            src_yr = int(float(papers[src]["year"]))
            dst_yr = int(float(papers[dst]["year"]))
            
            if dst_yr >= src_yr:
                dst_yr = src_yr - 1
                
            delta_ts.append(max(1, src_yr - dst_yr))

            
    # Assign edge data
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([cites_src, cites_dst], dtype=torch.long)
    data['paper', 'cites', 'paper'].edge_attr = torch.stack(edge_attrs)
    data['paper', 'cites', 'paper'].delta_t = torch.tensor(delta_ts, dtype=torch.float).unsqueeze(1)
    data['paper', 'cites', 'paper'].intent_label = torch.tensor(intent_labels, dtype=torch.long)
    data['paper', 'cites', 'paper'].train_mask = torch.tensor(label_masks, dtype=torch.bool)
    
    # Assign empty author-writes-paper data
    data['author', 'writes', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)

    print("\n✅ Constructed Scaled HeteroData:")
    print(data)
    
    # --- PHASE 1.5: NETWORK SANITY CHECK ---
    print("\n🧐 COMMENCING NETWORK SANITY CHECK...")
    labeled_count = data['paper', 'cites', 'paper'].train_mask.sum().item()
    total_edges = data['paper', 'cites', 'paper'].edge_index.shape[1]
    
    print(f"|--- Total Unique Papers (Nodes): {num_papers}")
    print(f"|--- Total Citation Edges (Links): {total_edges}")
    print(f"|    |--- Labeled Hop-1 Edges: {labeled_count}")
    print(f"|    |--- Unlabeled Hop-2 Edges: {total_edges - labeled_count}")
    
    # Check for density
    if num_papers > 0:
        avg_degree = total_edges / num_papers
        print(f"|--- Average Paper Citation Degree: {avg_degree:.2f}")

    print("✅ Sanity Check Passed. Proceeding to Heavy Vectorization...\n")

    # --- PHASE 2: CONSOLIDATION & EXPORT FOR DASHBOARD ---
    print("📦 Consolidation & Export for Dashboard...")
    
    # Merge all citation contexts into one Master file for the UI
    h1_ctx = hop1[['hop1_id', 'hop0_id', 'citation_text']].copy().rename(columns={'hop1_id':'citing_id', 'hop0_id':'cited_id', 'citation_text':'context'})
    h2_ctx = hop2[['hop2_id', 'hop1_id', 'citation_text']].copy().rename(columns={'hop2_id':'citing_id', 'hop1_id':'cited_id', 'citation_text':'context'})
    
    master_context = pd.concat([h1_ctx, h2_ctx]).dropna(subset=['context'])
    master_context.to_parquet("citation_contexts_scaled.parquet", index=False)
    
    # Save the master metadata dict for UI lookups (Title/Year)
    import json
    with open("hop01_metadata_scaled.json", "w") as f:
        json.dump(papers, f)
    
    # Save the actual Graph for Training
    out_path = "scicite_hetero_scaled.pt"
    torch.save(data, out_path)
    
    print(f"🎉 SUCCESS! Consolidate Artifacts Ready for Kaggle -> Local:")
    print(f" 1. Graph (Training): {out_path}")
    print(f" 2. Master Contexts (UI): citation_contexts_scaled.parquet")
    print(f" 3. Metadata Lookup (UI): hop01_metadata_scaled.json")

if __name__ == '__main__':
    main()
