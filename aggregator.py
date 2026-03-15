import sys
import json
import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ARXIV_FILE = "live_arxiv.jsonl"
S2ORC_FILE = "live_s2orc.jsonl"
CACHE_FILE = "aggregator_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(list(cache), f)

def fetch_arxiv(query):
    print(f"[arXiv] Thread started for query: '{query}'")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results=100"
    results = []
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            abstract_elem = entry.find('atom:summary', ns)
            pub_elem = entry.find('atom:published', ns)
            id_elem = entry.find('atom:id', ns)
            
            if title_elem is None or pub_elem is None:
                continue
                
            title = title_elem.text.replace('\n', ' ') if title_elem.text else ""
            abstract = abstract_elem.text.replace('\n', ' ') if (abstract_elem is not None and abstract_elem.text) else None
            published = pub_elem.text[:10]
            
            authors_parsed = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    name = name_elem.text
                    parts = name.split(' ')
                    if len(parts) >= 2:
                        authors_parsed.append([parts[-1], " ".join(parts[:-1]), ""])
                    else:
                        authors_parsed.append([name, "", ""])
                    
            results.append({
                "id": id_elem.text if id_elem is not None else "",
                "title": title,
                "abstract": abstract,
                "update_date": published,
                "authors_parsed": authors_parsed
            })
    except Exception as e:
        print(f"[arXiv] Error fetching '{query}': {e}")
    return results

def fetch_s2orc(query):
    print(f"[S2ORC] Thread started for query: '{query}'")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query.replace(' ', '+')}&limit=100&fields=title,abstract,authors,year"
    results = []
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'data' in data:
            for paper in data['data']:
                title = paper.get('title', '')
                abstract = paper.get('abstract', None)
                year = paper.get('year', 0)
                if not year:
                    year = 2023 # fallback
                    
                authors = []
                for author in paper.get('authors', []):
                    name = author.get('name', '')
                    if name:
                        parts = name.split(' ')
                        if len(parts) >= 2:
                            authors.append({"first": " ".join(parts[:-1]), "last": parts[-1]})
                        else:
                            authors.append({"first": name, "last": ""})
                
                results.append({
                    "paper_id": paper.get('paperId', ''),
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors
                })
    except Exception as e:
        print(f"[S2ORC] Error fetching '{query}': {e}")
    return results

def aggregate_query(query):
    arxiv_data = fetch_arxiv(query)
    s2orc_data = fetch_s2orc(query)
    
    if arxiv_data:
        with open(ARXIV_FILE, "a") as f:
            for item in arxiv_data:
                f.write(json.dumps(item) + "\n")
    
    if s2orc_data:
        with open(S2ORC_FILE, "a") as f:
            for item in s2orc_data:
                f.write(json.dumps(item) + "\n")
                
    print(f"✅ Fetched and saved {len(arxiv_data)} arXiv and {len(s2orc_data)} S2ORC papers for '{query}'.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregator.py \"query1\" \"query2\" ...")
        sys.exit()
        
    queries = sys.argv[1:]
    cache = load_cache()
    
    to_process = []
    for q in queries:
        if q.lower() in cache:
            print(f"⏩ Query '{q}' is already in cache. Skipping download.")
        else:
            to_process.append(q)
            
    if not to_process:
        print("All queries fulfilled from cache. Exiting.")
        return
        
    # Use ThreadPoolExecutor to fetch from sources in parallel
    print(f"Starting ThreadPoolExecutor for {len(to_process)} queries...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(aggregate_query, to_process)
        
    # Update cache
    for q in to_process:
        cache.add(q.lower())
    save_cache(cache)
    print("Done! Cache updated.")

if __name__ == "__main__":
    main()
