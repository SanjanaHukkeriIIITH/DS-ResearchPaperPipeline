"""
Load the crawled hop data into the edge table used by core.metrics.

The cleaned v2 CSVs have their Hop-0 / Hop-1 column names swapped:
    hop1_v2.hop1_id          -> SciCite *cited* paper   = true Hop-0
    hop1_v2.hop0_id          -> SciCite *citing* paper  = true Hop-1
    hop0_v2.source_paper_id  -> also the citing papers  = true Hop-1
    hop2_v2.hop1_id          -> the citing papers       = true Hop-1
    hop2_v2.hop2_id          -> papers citing those     = true Hop-2
(verified: citing papers are a median 6 years newer than cited ones).
Always load through this module rather than reading the CSVs directly.
"""

import os

import pandas as pd

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "cleaned")


def load_cleaned_v2(data_dir=DEFAULT_DIR):
    """Returns (edges, papers).

    edges : citing_id, cited_id, hop, intent, delta_t  (intent is the SciCite
            gold label on Hop-1 edges and NaN on Hop-2 edges until model
            predictions are merged in)
    papers: paper_id, layer, title, abstract, year
    """
    h1 = pd.read_csv(os.path.join(data_dir, "hop1_v2.csv"))
    hop1_meta = pd.read_csv(os.path.join(data_dir, "hop0_v2.csv"))
    h2 = pd.read_csv(os.path.join(data_dir, "hop2_v2.csv"))

    h1 = h1.rename(columns={
        "hop0_id": "citing_id", "hop1_id": "cited_id",
        "hop1_title": "cited_title", "hop1_abstract": "cited_abstract", "hop1_year": "cited_year",
    })
    h2 = h2.rename(columns={"hop1_id": "cited_id", "hop2_id": "citing_id", "hop2_year": "citing_year"})
    hop1_year = hop1_meta.set_index("source_paper_id")["year"]

    e1 = pd.DataFrame({
        "citing_id": h1["citing_id"],
        "cited_id": h1["cited_id"],
        "hop": 1,
        "intent": h1["intent_label"].str.lower().str.strip(),
        "delta_t": h1["citing_id"].map(hop1_year) - h1["cited_year"],
    })
    e2 = pd.DataFrame({
        "citing_id": h2["citing_id"],
        "cited_id": h2["cited_id"],
        "hop": 2,
        "intent": pd.NA,
        "delta_t": h2["citing_year"] - h2["cited_id"].map(hop1_year),
    })
    edges = pd.concat([e1, e2], ignore_index=True).drop_duplicates(["citing_id", "cited_id", "hop"])

    papers = pd.concat([
        h1[["cited_id", "cited_title", "cited_abstract", "cited_year"]]
            .drop_duplicates("cited_id")
            .set_axis(["paper_id", "title", "abstract", "year"], axis=1).assign(layer=0),
        hop1_meta.rename(columns={"source_paper_id": "paper_id"})
            [["paper_id", "title", "abstract", "year"]].assign(layer=1),
        h2[["citing_id", "hop2_title", "hop2_abstract", "citing_year"]]
            .drop_duplicates("citing_id")
            .set_axis(["paper_id", "title", "abstract", "year"], axis=1).assign(layer=2),
    ], ignore_index=True).drop_duplicates("paper_id")  # keep the lowest layer
    return edges, papers
