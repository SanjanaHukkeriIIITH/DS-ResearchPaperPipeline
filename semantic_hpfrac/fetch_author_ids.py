import requests
import time
import os
import json

TOP_30_AUTHORS = [
    "Anil K. Jain", "Yoshua Bengio", "Geoffrey Hinton", "Jiawei Han", 
    "Ian Goodfellow", "Andrew Ng", "Yann LeCun", "Philip S. Yu", 
    "Donald Knuth", "Kaiming He", "Tim Berners-Lee", "Jure Leskovec", 
    "David E. Goldberg", "Scott Shenker", "Michael I. Jordan", 
    "Hector Garcia-Molina", "Christos Papadimitriou", "Judea Pearl", 
    "Fei-Fei Li", "Ron Rivest", "Adi Shamir", "Jeffrey Dean", 
    "Sanjay Ghemawat", "Ilya Sutskever", "Vladimir Vapnik", 
    "Sebastian Thrun", "Jon Kleinberg", "Daphne Koller", 
    "Matei Zaharia", "Stuart Russell"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "New_expts", "top30_data")
CACHE_FILE = os.path.join(OUTPUT_DIR, "author_ids_cache.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def resolve_author_id(name):
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {"query": name, "limit": 1, "fields": "authorId,name"}
    headers = {"Accept": "application/json"}
    
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["authorId"]
                return None
            elif response.status_code == 429:
                time.sleep(5)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return None

def main():
    print("==================================================")
    print("🔍 FETCHING SEMANTIC SCHOLAR AUTHOR IDs")
    print("==================================================")
    
    author_cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            author_cache = json.load(f)
            
    for name in TOP_30_AUTHORS:
        if name in author_cache:
            print(f"✅ {name} -> {author_cache[name]} (cached)")
        else:
            print(f"⏳ Searching for {name}...")
            author_id = resolve_author_id(name)
            if author_id:
                author_cache[name] = author_id
                print(f"✅ Found: {author_id}")
            else:
                print(f"❌ Failed to resolve {name}")
            
            # Save incrementally
            with open(CACHE_FILE, "w") as f:
                json.dump(author_cache, f, indent=4)
                
            time.sleep(3) # Be nice to Semantic Scholar API
            
    print(f"\n🎉 Finished resolving {len(author_cache)}/{len(TOP_30_AUTHORS)} authors!")
    print(f"💾 Saved to: {CACHE_FILE}")

if __name__ == "__main__":
    main()
