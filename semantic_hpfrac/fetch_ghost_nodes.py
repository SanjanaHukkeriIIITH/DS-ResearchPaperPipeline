"""
fetch_ghost_nodes.py
────────────────────
Targeted metadata rescue for "Ghost Nodes" — Hop-0 paper IDs that are
referenced by Hop-1 papers but are absent from hop0_metadata_final.csv.

Reuses the same 3-tier fallback strategy from hop0_extract.py:
  1st: Semantic Scholar Batch API  (fast, covers most S2 IDs)
  2nd: Semantic Scholar Single API  (slower, sometimes has abstract when batch missed it)
  3rd: OpenAlex                      (good for older/non-S2 papers)
  4th: CrossRef                      (last resort for year + abstract)

Outputs:
  data/ghost_nodes_fetched.csv  → newly fetched ghost metadata
  data/hop0_metadata_final_v2.csv → original hop0 + ghosts merged together
"""

import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════
#  CONFIG — adjust as needed
# ══════════════════════════════════════════════════════════════
S2_API_KEY      = os.environ.get("S2_API_KEY", "")
BATCH_SIZE      = 500
SLEEP_S         = 1.5

S2_BATCH_URL    = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS          = "paperId,title,abstract,year"

CHECKPOINT_CSV  = "data/ghost_nodes_checkpoint.csv"
OUTPUT_CSV      = "data/ghost_nodes_fetched.csv"
HOP0_INPUT      = "data/hop0_metadata_final.csv"
HOP0_OUTPUT     = "data/hop0_metadata_final_v2.csv"   # original + ghosts merged

headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
# ══════════════════════════════════════════════════════════════


# ── Step 1: Identify the Ghost IDs ───────────────────────────
def get_ghost_ids() -> list:
    hop0 = pd.read_csv(HOP0_INPUT)
    hop1 = pd.read_csv("data/hop1_final_dataset_rescued.csv")

    valid_hop0_ids = set(hop0["source_paper_id"].dropna())
    cited_in_hop1  = set(hop1["hop0_id"].dropna())

    ghost_ids = list(cited_in_hop1 - valid_hop0_ids)
    print(f"✅ Identified {len(ghost_ids):,} Ghost Node IDs to fetch.")
    return ghost_ids


# ── API helpers (same as hop0_extract.py) ───────────────────

def fetch_s2_batch(ids: list, attempt: int = 1) -> list:
    resp = requests.post(
        S2_BATCH_URL,
        params={"fields": FIELDS},
        json={"ids": ids},
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 429:
        wait = 10 * attempt
        print(f"  Rate-limited (attempt {attempt}) — sleeping {wait}s …")
        time.sleep(wait)
        return fetch_s2_batch(ids, attempt + 1)
    resp.raise_for_status()
    return resp.json()


def try_s2_single(paper_id: str) -> dict:
    result = {"abstract": "", "year": None}
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            params={"fields": "abstract,year,title"},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            result["abstract"] = data.get("abstract") or ""
            result["year"]     = data.get("year")
            result["title"]    = data.get("title") or ""
    except Exception:
        pass
    return result


def try_openalex(title: str) -> dict:
    result = {"abstract": "", "year": None}
    if not title or str(title).strip() == "":
        return result
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": str(title).strip(), "per-page": 1,
                    "mailto": "research@example.com"},
            timeout=15,
        )
        if resp.status_code == 200:
            items = resp.json().get("results", [])
            if items:
                item = items[0]
                result["year"] = item.get("publication_year")
                inv = item.get("abstract_inverted_index")
                if inv:
                    words = {}
                    for word, positions in inv.items():
                        for pos in positions:
                            words[pos] = word
                    result["abstract"] = " ".join(words[i] for i in sorted(words))
    except Exception:
        pass
    return result


def try_crossref(title: str) -> dict:
    result = {"abstract": "", "year": None}
    if not title or str(title).strip() == "":
        return result
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.title": str(title).strip(), "rows": 1,
                    "mailto": "research@example.com"},
            timeout=15,
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                date_parts = item.get("issued", {}).get("date-parts", [])
                if date_parts and date_parts[0]:
                    result["year"] = date_parts[0][0]
                raw_abstract = item.get("abstract", "")
                if raw_abstract:
                    result["abstract"] = BeautifulSoup(
                        raw_abstract, "html.parser"
                    ).get_text(strip=True)
    except Exception:
        pass
    return result


# ── Main ─────────────────────────────────────────────────────
def main():
    all_ids = get_ghost_ids()
    total   = len(all_ids)

    # Resume checkpoint if it exists
    if os.path.exists(CHECKPOINT_CSV):
        ckpt_df      = pd.read_csv(CHECKPOINT_CSV)
        already_done = set(ckpt_df["source_paper_id"].tolist())
        records      = ckpt_df.to_dict("records")
        print(f"  Resuming checkpoint: {len(already_done):,} already fetched.")
    else:
        already_done = set()
        records      = []

    # ── Phase 1: S2 Batch fetch ──────────────────────────────
    print(f"\n📡 Phase 1 — S2 Batch fetch ({total:,} papers, batch={BATCH_SIZE})...")
    for start in range(0, total, BATCH_SIZE):
        batch = [pid for pid in all_ids[start: start + BATCH_SIZE]
                 if pid not in already_done]
        if not batch:
            continue

        print(f"  Batch {start}–{start+len(batch)-1} / {total} …")
        results = fetch_s2_batch(batch)

        for original_id, paper in zip(batch, results):
            if paper is None:
                records.append({
                    "source_paper_id": original_id,
                    "title": "", "abstract": "", "year": None, "source": "not_found"
                })
            else:
                # CRITICAL FIX: We must use the original_id as the key, 
                # otherwise the CSVs won't match during cleanup!
                records.append({
                    "source_paper_id": original_id,
                    "title":    paper.get("title")    or "",
                    "abstract": paper.get("abstract") or "",
                    "year":     paper.get("year"),
                    "source":   "s2_batch",
                })
            already_done.add(original_id)

        pd.DataFrame(records).to_csv(CHECKPOINT_CSV, index=False)
        time.sleep(SLEEP_S)

    print(f"  S2 Batch done — {len(records):,} records collected.")

    # ── Phase 2: Rescue loop (S2 single → OpenAlex → CrossRef) ─
    print(f"\n🔁 Phase 2 — Fallback rescue for missing abstract/year...")
    df = pd.DataFrame(records)
    needs_rescue = df[
        (df["abstract"].isna() | (df["abstract"].str.strip() == "")) |
        (df["year"].isna())
    ]
    print(f"  Papers needing rescue: {len(needs_rescue):,}")

    method_counts = {"s2_single": 0, "openalex": 0, "crossref": 0, "none": 0}

    for i, row in enumerate(needs_rescue.itertuples(), 1):
        pid   = row.source_paper_id
        title = str(row.title) if pd.notna(row.title) else ""

        need_abstract = not row.abstract or str(row.abstract).strip() == ""
        need_year     = pd.isna(row.year)

        filled_abstract = str(row.abstract) if not need_abstract else ""
        filled_year     = row.year if not need_year else None
        method = "none"

        # Tier 1: S2 single
        s2 = try_s2_single(pid)
        if need_abstract and s2.get("abstract", "").strip():
            filled_abstract = s2["abstract"]; need_abstract = False
        if need_year and s2.get("year"):
            filled_year = s2["year"]; need_year = False
        if not need_abstract and not need_year:
            method = "s2_single"; method_counts["s2_single"] += 1

        # Tier 2: OpenAlex
        if (need_abstract or need_year) and title:
            oa = try_openalex(title)
            if need_abstract and oa["abstract"].strip():
                filled_abstract = oa["abstract"]; need_abstract = False
            if need_year and oa["year"]:
                filled_year = oa["year"]; need_year = False
            if not need_abstract and not need_year:
                method = "openalex"; method_counts["openalex"] += 1
            time.sleep(1.0)

        # Tier 3: CrossRef
        if (need_abstract or need_year) and title:
            cr = try_crossref(title)
            if need_abstract and cr["abstract"].strip():
                filled_abstract = cr["abstract"]; need_abstract = False
            if need_year and cr["year"]:
                filled_year = cr["year"]; need_year = False
            if method == "none" and (filled_abstract or filled_year):
                method = "crossref"; method_counts["crossref"] += 1
            time.sleep(1.0)

        if method == "none":
            method_counts["none"] += 1

        # Update record in-place
        for rec in records:
            if rec["source_paper_id"] == pid:
                if filled_abstract: rec["abstract"] = filled_abstract
                if filled_year:     rec["year"]     = filled_year
                rec["source"] = method
                break

        pd.DataFrame(records).to_csv(CHECKPOINT_CSV, index=False)

        if i % 50 == 0:
            print(f"  [{i}/{len(needs_rescue)}] — "
                  f"s2:{method_counts['s2_single']} "
                  f"oa:{method_counts['openalex']} "
                  f"cr:{method_counts['crossref']} "
                  f"none:{method_counts['none']}")

    # ── Save fetched ghost data ──────────────────────────────
    ghost_df = pd.DataFrame(records)[["source_paper_id", "title", "year", "abstract", "source"]]
    ghost_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Ghost metadata saved → {OUTPUT_CSV}")

    # ── Merge with original hop0 and save combined v2 ────────
    hop0_orig = pd.read_csv(HOP0_INPUT)
    hop0_v2   = pd.concat([hop0_orig, ghost_df.drop(columns=["source"])], ignore_index=True)
    hop0_v2   = hop0_v2.drop_duplicates(subset=["source_paper_id"])
    hop0_v2.to_csv(HOP0_OUTPUT, index=False)
    print(f"✅ Merged Hop-0 saved    → {HOP0_OUTPUT}  ({len(hop0_v2):,} total seeds)")

    # ── Final Report ─────────────────────────────────────────
    has_abstract = ghost_df["abstract"].notna() & (ghost_df["abstract"].str.strip() != "")
    has_year     = ghost_df["year"].notna()
    print(f"\n{'═'*55}")
    print(f"  GHOST NODE RESCUE REPORT")
    print(f"{'═'*55}")
    print(f"  Ghost IDs targeted    : {total:,}")
    print(f"  With title+abstract   : {has_abstract.sum():,}  ({100*has_abstract.mean():.1f}%)")
    print(f"  With year             : {has_year.sum():,}  ({100*has_year.mean():.1f}%)")
    print(f"  Via s2_batch          : {sum(1 for r in records if r['source']=='s2_batch'):,}")
    print(f"  Via s2_single         : {method_counts['s2_single']:,}")
    print(f"  Via openalex          : {method_counts['openalex']:,}")
    print(f"  Via crossref          : {method_counts['crossref']:,}")
    print(f"  Still unknown         : {method_counts['none']:,}")
    print(f"{'═'*55}")
    print(f"\nNext step: run clean_hop_data.py using hop0_metadata_final_v2.csv")


if __name__ == "__main__":
    main()
