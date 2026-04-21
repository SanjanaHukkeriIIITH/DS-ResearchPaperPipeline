import pandas as pd
import json
import requests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

edges_df = pd.read_parquet("edge_predictions.parquet")
neighborhood_sizes = edges_df.groupby("cited_paper_id").size()
edges_df["relative_importance"] = edges_df["semantic_weight"] * edges_df["cited_paper_id"].map(neighborhood_sizes)

intent_multiplier = {"Background": 0.5, "Result": 1.0, "Method": 2.0, "Unknown": 1.0}
edges_df["intent_multi"] = edges_df["predicted_intent"].map(intent_multiplier)
edges_df["semantic_edge_score"] = edges_df["relative_importance"] * edges_df["intent_multi"]

paper_metrics = {}
for cited_id, group in edges_df.groupby("cited_paper_id"):
    paper_metrics[cited_id] = {
        "raw_citations": len(group),
        "semantic_citations": group["semantic_edge_score"].sum()
    }

cited_paper_ids = list(paper_metrics.keys())
url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,authors"
res = requests.post(url, json={"ids": cited_paper_ids})
author_mapping = res.json() if res.status_code == 200 else []

author_names = {}
author_metrics = defaultdict(lambda: {"papers": [], "normal_hp_frac": 0.0, "semantic_hp_frac": 0.0})

for original_pid, paper in zip(cited_paper_ids, author_mapping):
    if not paper or not paper.get("authors"):
        continue
    authors = paper["authors"]
    num_authors = max(1, len(authors))
    raw_cites = paper_metrics[original_pid]["raw_citations"]
    sem_cites = paper_metrics[original_pid]["semantic_citations"]
    for author in authors:
        aid = author.get("authorId")
        if not aid: continue
        author_names[aid] = author.get("name", f"Author_{aid}")
        author_metrics[aid]["papers"].append(raw_cites)
        author_metrics[aid]["normal_hp_frac"] += (raw_cites / num_authors)
        author_metrics[aid]["semantic_hp_frac"] += (sem_cites / num_authors)

for aid, data in author_metrics.items():
    sorted_cites = sorted(data["papers"], reverse=True)
    h_index = 0
    for i, c in enumerate(sorted_cites):
        if c >= (i + 1): h_index = i + 1
        else: break
    data["h_index"] = h_index

records = []
for aid, data in author_metrics.items():
    if data["h_index"] > 0:
        records.append({
            "Author Name": author_names.get(aid, "Unknown"),
            "H-Index": data["h_index"],
            "Normal": data["normal_hp_frac"],
            "Semantic": round(data["semantic_hp_frac"], 2)
        })

print(pd.DataFrame(records).sort_values(by="Semantic", ascending=False).head(5).to_string(index=False))
