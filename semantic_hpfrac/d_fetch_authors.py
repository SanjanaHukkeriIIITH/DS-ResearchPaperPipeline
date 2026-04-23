import pandas as pd
import requests
import json
import os
import time
import tqdm

# ── Fallback Helpers (Modeled after your ghost node rescue) ──

def try_openalex_authors(title):
    """Fallback 1: Query OpenAlex by Title for Authors"""
    if not title or str(title).strip() == "" or str(title).lower() == "nan":
        return []
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": str(title).strip(), "per-page": 1, "mailto": "research@example.com"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("results", [])
            if items:
                authorships = items[0].get("authorships", [])
                authors = []
                for a in authorships:
                    auth_data = a.get("author", {})
                    auth_id = auth_data.get("id")
                    auth_name = auth_data.get("display_name", "Unknown")
                    if auth_id:
                        # OpenAlex IDs look like 'https://openalex.org/A1234', we just want the 'A1234'
                        clean_id = auth_id.split("/")[-1]
                        authors.append({"author_id": f"oa_{clean_id}", "author_name": auth_name})
                return authors
    except Exception:
        pass
    return []

def try_crossref_authors(title):
    """Fallback 2: Query CrossRef by Title for Authors"""
    if not title or str(title).strip() == "" or str(title).lower() == "nan":
        return []
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.title": str(title).strip(), "rows": 1, "mailto": "research@example.com"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            if items:
                raw_authors = items[0].get("author", [])
                authors = []
                for a_idx, a in enumerate(raw_authors):
                    # CrossRef uses ORCID, but it's often missing. If missing, generate a name-based ID.
                    auth_id = a.get("ORCID")
                    name = f"{a.get('given', '')} {a.get('family', '')}".strip() or "Unknown"
                    
                    if auth_id:
                        clean_id = auth_id.replace("http://orcid.org/", "")
                        authors.append({"author_id": f"cr_{clean_id}", "author_name": name})
                    else:
                        clean_name = name.lower().replace(" ", "_")
                        authors.append({"author_id": f"cr_{clean_name}_{a_idx}", "author_name": name})
                return authors
    except Exception:
        pass
    return []

# ── Main Pipeline ──

def main():
    print("1. Extracting unique Paper IDs and Titles from all Hop CSVs...")
    hop0 = pd.read_csv("data/cleaned/hop0_v2.csv")
    hop1 = pd.read_csv("data/cleaned/hop1_v2.csv")
    hop2 = pd.read_csv("data/cleaned/hop2_v2.csv")

    # We need a dictionary mapping ID -> Title for the fallbacks
    id_to_title = {}
    for _, row in hop0.iterrows():
        id_to_title[str(row['source_paper_id']).strip()] = str(row['title'])
    for _, row in hop1.iterrows():
        id_to_title[str(row['hop1_id']).strip()] = str(row['hop1_title'])
    for _, row in hop2.iterrows():
        id_to_title[str(row['hop2_id']).strip()] = str(row['hop2_title'])

    all_paper_ids = list(id_to_title.keys())
    print(f"Total unique papers to query: {len(all_paper_ids):,}")

    output_file = "data/paper_authors_edges.csv"
    checkpoint_file = "data/authors_checkpoint.txt"
    
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_ids = set(f.read().splitlines())
            
    remaining_papers = [pid for pid in all_paper_ids if pid not in processed_ids]
    print(f"Resuming pipeline. {len(remaining_papers):,} papers remaining to fetch.")

    print("\n2. Commencing Multi-Tier Author Extraction...")
    api_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    query_params = "?fields=authors.authorId,authors.name"
    headers = {'Content-Type': 'application/json'} 
    
    batch_size = 100
    author_edges = []

    if not os.path.exists(output_file):
        pd.DataFrame(columns=["paper_id", "author_id", "author_name", "source"]).to_csv(output_file, index=False)

    for i in tqdm.tqdm(range(0, len(remaining_papers), batch_size)):
        batch_ids = remaining_papers[i:i+batch_size]
        payload = json.dumps({"ids": batch_ids})
        
        try:
            response = requests.post(api_url + query_params, headers=headers, data=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                for idx, paper_data in enumerate(data):
                    src_id = batch_ids[idx]
                    paper_title = id_to_title.get(src_id, "")
                    extracted_for_paper = False
                    
                    # TIER 1: Semantic Scholar
                    if paper_data and paper_data.get('authors'):
                        for a_idx, author in enumerate(paper_data['authors']):
                            auth_id = author.get('authorId')
                            if not auth_id:
                                auth_id = f"s2_unknown_{src_id}_{a_idx}"
                            author_edges.append({
                                "paper_id": src_id, "author_id": str(auth_id), 
                                "author_name": author.get('name', 'Unknown'), "source": "S2"
                            })
                        extracted_for_paper = True
                        
                    # TIER 2: OpenAlex Fallback
                    if not extracted_for_paper:
                        oa_authors = try_openalex_authors(paper_title)
                        if oa_authors:
                            for a in oa_authors:
                                a.update({"paper_id": src_id, "source": "OpenAlex"})
                                author_edges.append(a)
                            extracted_for_paper = True
                            
                    # TIER 3: CrossRef Fallback
                    if not extracted_for_paper:
                        cr_authors = try_crossref_authors(paper_title)
                        if cr_authors:
                            for a in cr_authors:
                                a.update({"paper_id": src_id, "source": "CrossRef"})
                                author_edges.append(a)
                            extracted_for_paper = True
                            
                    # TIER 4: The Ultimate Dummy Safety Net
                    if not extracted_for_paper:
                        author_edges.append({
                            "paper_id": src_id, 
                            "author_id": f"final_unknown_{src_id}_0", 
                            "author_name": "Unknown", 
                            "source": "Dummy_Gen"
                        })
                
                # Checkpointing logic
                if author_edges:
                    pd.DataFrame(author_edges).to_csv(output_file, mode='a', header=False, index=False)
                    author_edges = []
                    
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

if __name__ == '__main__':
    main()