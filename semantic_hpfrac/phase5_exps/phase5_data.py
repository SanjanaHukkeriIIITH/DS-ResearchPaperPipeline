"""
Shared input for every Phase 5 script.

Builds ONE edge table and ONE authorship table, so every ablation, sweep and
attack runs on exactly the same graph. All metric maths lives in core/metrics.py;
nothing here computes a score.

Sources
  - graph            : data/cleaned/cleaned/*_v2.csv via core.loaders (Hop-0/Hop-1 swap fixed)
  - Hop-1 intent     : SciCite gold labels (already in the loader output)
  - Hop-2 intent, alpha : PREDICTIONS FILE (default: the old April run,
                       data/data_v2/edge_predictions_scaled.parquet). These are
                       stand-ins until the Phase 1 retrain; swap the file with
                       --predictions and rerun. Nothing else changes.
  - authorship       : data/paper_authors_edges.csv, restricted to Hop-0 papers

Edges kept (same set for every variant, so differences come only from the config):
  - Hop-1 edges with a known time gap
  - Hop-2 edges with a known time gap AND a prediction (intent + alpha)
Hop-1 edges have no model attention, so alpha is set uniform on them; under the
rescaling in core.metrics that makes the attention term exactly 1 (no effect).
alpha on Hop-2 edges is re-normalised over the in-edges that survive the filter.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # semantic_hpfrac/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.loaders import load_cleaned_v2         # noqa: E402

DEFAULT_PREDICTIONS = os.path.join(ROOT, "data", "data_v2", "edge_predictions_scaled.parquet")
DEFAULT_AUTHORS = os.path.join(ROOT, "data", "paper_authors_edges.csv")
REPORTS = os.path.join(HERE, "reports")

BANNER = (
    "NOTE: Hop-2 intents/attention come from the OLD model run and lambda is a\n"
    "placeholder until Phase 1 finishes. These numbers test the pipeline only;\n"
    "do not put them in the paper. Rerun with --predictions / --lam afterwards."
)


def uniform_alpha(edges):
    return 1.0 / edges.groupby("cited_id")["citing_id"].transform("size")


def load_phase5_graph(predictions=DEFAULT_PREDICTIONS, authors=DEFAULT_AUTHORS):
    """Returns (edges, authorship, info). info is a dict of counts for the report."""
    edges, _ = load_cleaned_v2()
    info = {"hop1_edges_raw": int((edges["hop"] == 1).sum()),
            "hop2_edges_raw": int((edges["hop"] == 2).sum())}

    pred = pd.read_parquet(predictions).rename(
        columns={"citing_paper_id": "citing_id", "cited_paper_id": "cited_id",
                 "predicted_intent": "pred_intent", "raw_importance": "alpha"})
    pred = pred[["citing_id", "cited_id", "pred_intent", "alpha"]].drop_duplicates(["citing_id", "cited_id"])
    pred["pred_intent"] = pred["pred_intent"].str.lower().str.strip()

    h1 = edges[(edges["hop"] == 1) & edges["delta_t"].notna()].copy()
    h2 = edges[(edges["hop"] == 2) & edges["delta_t"].notna()].drop(columns="intent")
    h2 = h2.merge(pred, on=["citing_id", "cited_id"], how="inner").rename(columns={"pred_intent": "intent"})

    h1["alpha"] = uniform_alpha(h1)
    h2["alpha"] = h2["alpha"] / h2.groupby("cited_id")["alpha"].transform("sum")

    edges = pd.concat([h1, h2], ignore_index=True)
    edges["intent"] = edges["intent"].astype(str)
    citers = h1.groupby("cited_id").size()
    attn = h2["alpha"] * h2.groupby("cited_id")["citing_id"].transform("size")
    info.update(hop1_edges_used=len(h1), hop2_edges_used=len(h2),
                hop0_papers=int(h1["cited_id"].nunique()),
                share_hop0_with_one_citer=float((citers == 1).mean()),
                max_hop2_citers=int(h2.groupby("cited_id").size().max()),
                attention_spread=float(attn.max() - attn.min()))

    hop0 = set(h1["cited_id"])
    auth = pd.read_csv(authors, dtype={"paper_id": str, "author_id": str})
    auth = auth[auth["paper_id"].isin(hop0)].drop_duplicates(["paper_id", "author_id"])
    authorship = auth[["paper_id", "author_id"]].reset_index(drop=True)
    info.update(authors=int(authorship["author_id"].nunique()),
                hop0_papers_with_authors=int(authorship["paper_id"].nunique()),
                max_hop0_papers_per_author=int(authorship.groupby("author_id").size().max()))
    return edges, authorship, info


def rank_agreement(a, b, top_k=100):
    """Kendall tau-b between two score Series (aligned on index, missing = 0) and
    the overlap of their top-k sets. Ties are common with h-type scores, so the
    top-k set is taken by sorting on score then id, and tau-b handles ties."""
    from scipy.stats import kendalltau
    idx = a.index.union(b.index)
    a, b = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    tau = kendalltau(a.to_numpy(), b.to_numpy()).statistic
    k = min(top_k, len(idx))

    def top(s):
        d = pd.DataFrame({"id": s.index.astype(str), "v": s.to_numpy()})
        return set(d.sort_values(["v", "id"], ascending=[False, True])["id"].iloc[:k])
    return float(tau), len(top(a) & top(b)) / k if k else float("nan")


def describe(info):
    lines = [
        f"Graph: {info['hop0_papers']:,} Hop-0 papers | Hop-1 edges {info['hop1_edges_used']:,}/"
        f"{info['hop1_edges_raw']:,} | Hop-2 edges {info['hop2_edges_used']:,}/{info['hop2_edges_raw']:,} (kept/total)",
        f"Authors: {info['authors']:,} on {info['hop0_papers_with_authors']:,} Hop-0 papers "
        f"(max {info['max_hop0_papers_per_author']} papers per author)",
    ]
    warn = []
    if info["share_hop0_with_one_citer"] > 0.5:
        warn.append(f"{info['share_hop0_with_one_citer']:.1%} of Hop-0 papers have exactly ONE Hop-1 citer here "
                    "(SciCite lists one citing paper per pair), so 2-hop paper scores can only be 0, 1 or 2.")
    if info["attention_spread"] < 0.2:
        warn.append(f"Attention is near-uniform in this predictions file (spread {info['attention_spread']:.3f}), "
                    "so attention variants cannot differ from their no-attention twins.")
    if info["max_hop0_papers_per_author"] < 3:
        warn.append("No author has 3+ Hop-0 papers, so author-level results are just paper-level ones divided by a_p.")
    if warn:
        lines.append("DATA LIMITS (results below are a pipeline test, not findings):")
        lines += [f"  - {w}" for w in warn]
    return "\n".join(lines)
