import requests
import time
import os
import pandas as pd
import json
import urllib.parse
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Target Author ID (Andrej Karpathy = 2354728)
AUTHOR_ID = "2354728"

# S2 API Key - highly recommended but optional
S2_API_KEY = None 

# Directories for Checkpointing
OUTPUT_DIR = "karpathy_exhaustive_network"
HOP0_FILE = f"{OUTPUT_DIR}/hop0_data.csv"
HOP1_FILE = f"{OUTPUT_DIR}/hop1_data.csv"
HOP2_FILE = f"{OUTPUT_DIR}/hop2_data.csv"
PROGRESS_FILE = f"{OUTPUT_DIR}/crawl_progress.json"

# Delay to prevent 429 Too Many Requests (in seconds)
BASE_DELAY = 3.0  

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# ROBUST API SESSION WITH EXPONENTIAL BACKOFF
# ==============================================================================
def robust_s2_request(url, params=None):
    headers = {"Accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
        
    backoff = BASE_DELAY
    max_retries = 10
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                time.sleep(BASE_DELAY) 
                return response.json()
                
            elif response.status_code == 429:
                # S2 free tier limits reset every 5 minutes (300 seconds).
                # We check the Retry-After header, or default to a 5-minute wait.
                retry_after = int(response.headers.get("Retry-After", 305))
                sleep_time = max(backoff, retry_after)
                print(f"  [!] 429 Rate Limit Hit. Sleeping for {sleep_time} seconds to clear quota...")
                time.sleep(sleep_time)
                # Next backoff increases just in case, but usually we just wait the retry_after limit
                if sleep_time == backoff:
                    backoff *= 2
                else:
                    backoff = retry_after + 5 # add buffer for next time
                
            elif response.status_code == 404:
                print("  [!] Resource not found.")
                return None
                
            else:
                print(f"  [!] Unexpected Status {response.status_code}. Retrying...")
                time.sleep(5)
                
        except Exception as e:
            print(f"  [!] Connection Error: {e}. Sleeping 10s...")
            time.sleep(10)
            
    print("  [!] Max retries exceeded for URL:", url)
    return None

def fetch_s2_details(paper_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "paperId,title,year,authors,abstract,externalIds"}
    data = robust_s2_request(url, params)
    return data if data else {}

def fetch_from_openalex(title, doi):
    headers = {"Accept": "application/json"}
    try:
        if doi:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = requests.get(url, headers=headers, timeout=15)
        else:
            if not title or title == "N/A": return {}
            url = f"https://api.openalex.org/works?filter=title.search:{title}"
            response = requests.get(url, headers=headers, timeout=15)
            
        if response.status_code == 200:
            data = response.json()
            if not doi:
                if data.get("results") and len(data["results"]) > 0:
                    work = data["results"][0]
                else:
                    return {}
            else:
                work = data
                
            abstract = work.get("abstract_inverted_index")
            abstract_text = ""
            if abstract:
                words = []
                for word, positions in abstract.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort()
                abstract_text = " ".join([w[1] for w in words])
                
            authors = work.get("authorships", [])
            return {
                "abstract": abstract_text,
                "year": work.get("publication_year"),
                "num_authors": len(authors) if authors else 0
            }
    except Exception as e:
        pass
    return {}

def fetch_from_crossref(title, doi):
    headers = {"Accept": "application/json"}
    try:
        if doi:
            url = f"https://api.crossref.org/works/{doi}"
        else:
            if not title or title == "N/A": return {}
            encoded_title = urllib.parse.quote(title)
            url = f"https://api.crossref.org/works?query.title={encoded_title}&rows=1"
            
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            msg = data.get("message", {})
            if not doi:
                items = msg.get("items", [])
                if not items: return {}
                work = items[0]
            else:
                work = msg
                
            abstract_raw = work.get("abstract", "")
            abstract_text = re.sub('<[^<]+>', '', abstract_raw) if abstract_raw else ""
            
            authors = work.get("author", [])
            year = None
            published = work.get("published-print") or work.get("published-online")
            if published and "date-parts" in published:
                parts = published["date-parts"]
                if parts and len(parts[0]) > 0:
                    year = parts[0][0]
                    
            return {
                "abstract": abstract_text,
                "year": year,
                "num_authors": len(authors) if authors else 0
            }
    except Exception as e:
        pass
    return {}

def extract_paper_data(paper_data):
    if not paper_data: 
        return {"title": "N/A", "year": "N/A", "num_authors": 0, "abstract": "N/A", "raw_abstract": ""}
        
    paper_id = paper_data.get("paperId")
    title = paper_data.get("title")
    title_str = str(title) if title else "N/A"
    
    authors = paper_data.get("authors") or []
    year = paper_data.get("year")
    raw_abstract = paper_data.get("abstract") or ""
    
    externalIds = paper_data.get("externalIds", {})
    doi = externalIds.get("DOI")
    
    needs_fallback = not raw_abstract or year is None
    if needs_fallback and paper_id:
        # 1. Semantic Scholar Deep Fetch
        s2_deep = fetch_s2_details(paper_id)
        if s2_deep:
            raw_abstract = raw_abstract or s2_deep.get("abstract") or ""
            if year is None: year = s2_deep.get("year")
            if not authors: authors = s2_deep.get("authors") or []
            if not doi: doi = s2_deep.get("externalIds", {}).get("DOI")
            
        # 2. OpenAlex
        needs_fallback = not raw_abstract or year is None
        if needs_fallback:
            oa_data = fetch_from_openalex(title_str, doi)
            if oa_data:
                raw_abstract = raw_abstract or oa_data.get("abstract") or ""
                if year is None: year = oa_data.get("year")
                if not authors and oa_data.get("num_authors", 0) > 0:
                    authors = [1] * oa_data.get("num_authors")
                    
        # 3. Crossref
        needs_fallback = not raw_abstract or year is None
        if needs_fallback:
            cr_data = fetch_from_crossref(title_str, doi)
            if cr_data:
                raw_abstract = raw_abstract or cr_data.get("abstract") or ""
                if year is None: year = cr_data.get("year")
                if not authors and cr_data.get("num_authors", 0) > 0:
                    authors = [1] * cr_data.get("num_authors")

    year_str = str(year) if year is not None else "N/A"
    abstract_str = raw_abstract if raw_abstract else title_str
        
    return {
        "title": title_str,
        "year": year_str,
        "num_authors": len(authors),
        "abstract": abstract_str,
        "raw_abstract": raw_abstract
    }

# ==============================================================================
# CRAWL LOGIC WITH CHECKPOINTING
# ==============================================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"processed_hop0": [], "processed_hop1": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)

def fetch_citations(paper_id):
    all_citations = []
    offset = 0
    limit = 1000 
    fields = "contexts,citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.authors,citingPaper.abstract,citingPaper.externalIds"
    
    while True:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = {"fields": fields, "offset": offset, "limit": limit}
        
        data = robust_s2_request(url, params)
        if not data or "data" not in data: break
            
        citations = data["data"]
        all_citations.extend(citations)
        
        if len(citations) < limit: break 
        offset += limit
        
    return all_citations

def main():
    print("=" * 80)
    print(f"🕸️  ROBUST EXHAUSTIVE CRAWLER: ENTIRE GRAPH FOR AUTHOR {AUTHOR_ID}")
    print("=" * 80)
    progress = load_progress()
    
    # -------------------------------------------------------------------------
    # PHASE 0: CRAWL ALL HOP-0 PAPERS FOR THE AUTHOR
    # -------------------------------------------------------------------------
    print("\n--- PHASE 0: Fetching ALL Author Papers (Hop-0) ---")
    
    if not os.path.exists(HOP0_FILE):
        url = f"https://api.semanticscholar.org/graph/v1/author/{AUTHOR_ID}/papers"
        params = {"fields": "paperId,title,year,authors,abstract,externalIds", "limit": 1000}
        data = robust_s2_request(url, params)
        
        if not data or "data" not in data:
            print("Failed to fetch author papers.")
            return
            
        hop0_records = []
        for paper in data["data"]:
            if not paper.get("paperId"): continue
            ext = extract_paper_data(paper)
            hop0_records.append({
                "hop0_paper_id": paper["paperId"],
                "title": ext["title"],
                "year": ext["year"],
                "num_authors": ext["num_authors"],
                "abstract": ext["abstract"],
                "citation_contexts": "N/A"
            })
            
        df_hop0 = pd.DataFrame(hop0_records)
        df_hop0.to_csv(HOP0_FILE, index=False)
        print(f"  ✅ Saved {len(hop0_records)} Hop-0 papers to {HOP0_FILE}.")
    else:
        df_hop0 = pd.read_csv(HOP0_FILE)
        print(f"  ✅ Loaded {len(df_hop0)} Hop-0 papers from cache.")

    # -------------------------------------------------------------------------
    # PHASE 1: CRAWL HOP-1 (Papers citing ALL Hop-0 papers)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: Crawling Hop-1 Citations ---")
    hop0_ids = df_hop0["hop0_paper_id"].unique()
    total_hop0 = len(hop0_ids)
    
    for i, pid in enumerate(hop0_ids):
        if pid in progress["processed_hop0"]:
            continue
            
        title = df_hop0[df_hop0['hop0_paper_id'] == pid]['title'].iloc[0]
        print(f"[{i+1}/{total_hop0}] 📥 Fetching Hop-1 for: {title[:50]}...")
        
        citations = fetch_citations(pid)
        new_records = []
        for cite in citations:
            citing_paper = cite.get("citingPaper", {})
            if not citing_paper.get("paperId"): continue
            
            contexts = cite.get("contexts", [])
            ext = extract_paper_data(citing_paper)
            if contexts:
                citation_contexts = " | ".join(contexts)
            elif ext["raw_abstract"]:
                citation_contexts = ext["raw_abstract"]
            else:
                citation_contexts = "N/A"
                
            new_records.append({
                "hop0_seed": pid,
                "hop1_paper_id": citing_paper["paperId"],
                "title": ext["title"],
                "year": ext["year"],
                "num_authors": ext["num_authors"],
                "abstract": ext["abstract"],
                "citation_contexts": citation_contexts
            })
            
        if new_records:
            pd.DataFrame(new_records).to_csv(HOP1_FILE, mode='a', header=not os.path.exists(HOP1_FILE), index=False)
            
        progress["processed_hop0"].append(pid)
        save_progress(progress)
        print(f"  ✅ Saved {len(new_records)} citations.")

    # -------------------------------------------------------------------------
    # PHASE 2: CRAWL HOP-2 (Papers citing the Hop-1 papers)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: Crawling Hop-2 Citations ---")
    if not os.path.exists(HOP1_FILE):
        print("No Hop-1 data found. Exiting.")
        return
        
    df_hop1 = pd.read_csv(HOP1_FILE)
    unique_hop1_ids = df_hop1["hop1_paper_id"].dropna().unique()
    total_hop1 = len(unique_hop1_ids)
    
    for i, h1_pid in enumerate(unique_hop1_ids):
        if h1_pid in progress["processed_hop1"]:
            continue
            
        print(f"[{i+1}/{total_hop1}] 📥 Fetching Hop-2 for Hop-1 Paper: {h1_pid}")
        citations = fetch_citations(h1_pid)
        
        new_hop2_records = []
        for cite in citations:
            citing_paper = cite.get("citingPaper", {})
            if not citing_paper.get("paperId"): continue
            
            contexts = cite.get("contexts", [])
            ext = extract_paper_data(citing_paper)
            if contexts:
                citation_contexts = " | ".join(contexts)
            elif ext["raw_abstract"]:
                citation_contexts = ext["raw_abstract"]
            else:
                citation_contexts = "N/A"
                
            new_hop2_records.append({
                "hop1_seed": h1_pid,
                "hop2_paper_id": citing_paper["paperId"],
                "title": ext["title"],
                "year": ext["year"],
                "num_authors": ext["num_authors"],
                "abstract": ext["abstract"],
                "citation_contexts": citation_contexts
            })
            
        if new_hop2_records:
            pd.DataFrame(new_hop2_records).to_csv(HOP2_FILE, mode='a', header=not os.path.exists(HOP2_FILE), index=False)
            
        progress["processed_hop1"].append(h1_pid)
        save_progress(progress)

    print("\n✅ Exhaustive Crawl Complete!")

if __name__ == "__main__":
    main()
