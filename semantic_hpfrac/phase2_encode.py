import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import json

# ==============================================================================
# PHASE 2: SCI-BERT VECTORIZATION & METADATA ENRICHMENT
# ==============================================================================

S2_API_KEY = None  # Add if you have one
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def fetch_s2_metadata(paper_ids):
    """Fetch Title, Year, and Abstract in batches of 100."""
    metadata = {}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,title,year,abstract"
    headers = {"Accept": "application/json"}
    if S2_API_KEY: headers["x-api-key"] = S2_API_KEY
    
    unique_ids = list(set(paper_ids))
    for i in tqdm(range(0, len(unique_ids), 100), desc="Fetching missing metadata from S2"):
        batch = unique_ids[i:i+100]
        try:
            res = requests.post(url, headers=headers, json={"ids": batch})
            if res.status_code == 200:
                for paper in res.json():
                    if paper and paper.get("paperId"):
                        metadata[paper["paperId"]] = {
                            "title": paper.get("title", ""),
                            "year": paper.get("year", ""),
                            "abstract": paper.get("abstract", "")
                        }
        except Exception as e:
            print(f"Batch failed: {e}")
    return metadata

def main():
    print("Loading graph data...")
    hop1_df = pd.read_parquet("scicite_training_data.parquet")
    hop2_df = pd.read_parquet("hop2_edges.parquet")
    
    # 1. Gather all unique IDs in our active cluster
    active_hop1_ids = set(hop2_df["hop1_paper_id"].dropna().unique())
    sampled_hop1_df = hop1_df[hop1_df["citingPaperId"].isin(active_hop1_ids)].copy()
    active_hop0_ids = set(sampled_hop1_df["citedPaperId"].dropna().unique())
    
    print(f"Hop-0 Seeds: {len(active_hop0_ids)}")
    print(f"Hop-1 Nodes: {len(active_hop1_ids)}")
    print(f"Hop-2 Nodes: {len(hop2_df)}")
    
    # 2. Fetch missing Text Metadata for Hop-0 and Hop-1
    ids_to_fetch = list(active_hop0_ids.union(active_hop1_ids))
    meta_path = "hop01_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            hop01_metadata = json.load(f)
        print("Loaded cached S2 metadata.")
    else:
        hop01_metadata = fetch_s2_metadata(ids_to_fetch)
        with open(meta_path, "w") as f:
            json.dump(hop01_metadata, f)
            
    # 3. Assemble Rich Text strings for each node
    # We will build a dictionary: paper_id -> Rich Text String
    node_texts = {}
    
    # Build text for Hop-0 (Seeds)
    for pid in active_hop0_ids:
        meta = hop01_metadata.get(pid, {})
        t = meta.get("title", "")
        y = meta.get("year", "")
        a = meta.get("abstract", "")
        if t or a:
            node_texts[pid] = f"Title: {t}. Year: {y}. Abstract: {a}"
        else:
            node_texts[pid] = "Empty Seed Node"

    # Build text for Hop-1 (Citations with their context towards Hop-0)
    for _, row in sampled_hop1_df.iterrows():
        h1 = row["citingPaperId"]
        meta = hop01_metadata.get(h1, {})
        t = meta.get("title", "")
        y = meta.get("year", "")
        a = meta.get("abstract", "")
        
        # Inject the Scicite metadata (citation sentence & section)
        sec = row.get("sectionName", "")
        cit_sentence = row.get("string", "")
        
        node_texts[h1] = f"Title: {t}. Year: {y}. Citation to Target [{sec}]: '{cit_sentence}'. Abstract: {a}"

    # Build text for Hop-2 (Citations referencing Hop-1)
    for _, row in hop2_df.iterrows():
        h2 = row["hop2_paper_id"]
        if h2 not in node_texts: # In case Hop-2 cites multiple Hop-1s, just encode once
            t = row.get("hop2_title", "")
            y = row.get("hop2_year", "")
            a = row.get("hop2_abstract", "")
            cit_sentence = row.get("citation_context", "")
            
            node_texts[h2] = f"Title: {t}. Year: {y}. Citation Context: '{cit_sentence}'. Abstract: {a}"
            
    print(f"Total Nodes prepared for SciBERT: {len(node_texts):,}")
    
    # 4. SciBERT Vectorization
    print("Loading SciBERT...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(DEVICE)
    model.eval()
    
    embeddings = {}
    
    # Iterate through dictionary
    items = list(node_texts.items())
    print("Encoding nodes via SciBERT. This may take a moment...")
    
    # Process in small mini-batches to fit in GPU memory
    batch_size = 16 
    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size), desc="SciBERT Encoding"):
            batch = items[i:i+batch_size]
            batch_ids = [k for k,v in batch]
            batch_texts = [str(v) if v else "" for k,v in batch]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)
            
            # Use the [CLS] token representation for the whole document embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            
            for j, pid in enumerate(batch_ids):
                embeddings[pid] = cls_embeddings[j].clone()
                
    # 5. Save the embeddings
    torch.save(embeddings, "node_embeddings.pt")
    print(f"\\n✅ Successfully saved 768-D embeddings for {len(embeddings)} nodes to 'node_embeddings.pt'!")

if __name__ == "__main__":
    main()
