import pandas as pd
import requests
import json
import os
import time
import tqdm

def main():
    print("1. Extracting unique Paper IDs from all Hop CSVs...")
    # Load your cleaned datasets
    hop0 = pd.read_csv("data/cleaned/cleaned/hop0_v2.csv")
    hop1 = pd.read_csv("data/cleaned/cleaned/hop1_v2.csv")
    hop2 = pd.read_csv("data/cleaned/cleaned/hop2_v2.csv")

    # Gather every unique paper that exists in our graph
    p0 = set(hop0['source_paper_id'].dropna().astype(str).str.strip())
    p1 = set(hop1['hop1_id'].dropna().astype(str).str.strip())
    p2 = set(hop2['hop2_id'].dropna().astype(str).str.strip())
    
    all_paper_ids = list(p0 | p1 | p2)
    print(f"Total unique papers to query: {len(all_paper_ids):,}")

    # Setup Checkpointing
    output_file = "data/paper_authors_edges.csv"
    checkpoint_file = "data/authors_checkpoint.txt"
    
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_ids = set(f.read().splitlines())
            
    remaining_papers = [pid for pid in all_paper_ids if pid not in processed_ids]
    print(f"Resuming pipeline. {len(remaining_papers):,} papers remaining to fetch.")

    print("\n2. Fetching Author Data via Semantic Scholar API...")
    api_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    query_params = "?fields=authors.authorId,authors.name"
    # Note: If you have an S2 API key, add it to the headers: {"x-api-key": "YOUR_KEY"}
    headers = {'Content-Type': 'application/json'} 
    
    batch_size = 100
    author_edges = []

    # Write headers if starting fresh
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["paper_id", "author_id", "author_name"]).to_csv(output_file, index=False)

    for i in tqdm.tqdm(range(0, len(remaining_papers), batch_size)):
        batch_ids = remaining_papers[i:i+batch_size]
        payload = json.dumps({"ids": batch_ids})
        
        try:
            response = requests.post(api_url + query_params, headers=headers, data=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                for idx, paper_data in enumerate(data):
                    if paper_data and 'authors' in paper_data:
                        src_id = batch_ids[idx]
                        
                        # Extract each author for this paper
                        # Extract each author for this paper
                        for a_idx, author in enumerate(paper_data['authors']):
                            auth_id = author.get('authorId')
                            
                            # Generate a unique dummy ID if missing so the a_i count stays accurate!
                            if not auth_id:
                                auth_id = f"unknown_{src_id}_{a_idx}"
                                
                            author_edges.append({
                                "paper_id": src_id,
                                "author_id": str(auth_id),
                                "author_name": author.get('name', 'Unknown')
                            })
                
                # Append to CSV and update checkpoint
                if author_edges:
                    pd.DataFrame(author_edges).to_csv(output_file, mode='a', header=False, index=False)
                    author_edges = [] # clear RAM
                    
                with open(checkpoint_file, 'a') as f:
                    for pid in batch_ids:
                        f.write(f"{pid}\n")
                        
            elif response.status_code == 429:
                print("\nRate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
                
        except Exception as e:
            print(f"\nNetwork timeout on batch {i}. Retrying in 5s...")
            time.sleep(5)

    print("\n✅ Author Extraction Complete!")
    print(f"Data successfully saved to {output_file}")

if __name__ == '__main__':
    main()