#!/usr/bin/env python3
"""
inject_hop2_cells.py
--------------------
Programmatically appends two new cells to hpfrac.ipynb:
  1. A markdown header cell explaining the Phase 1 / Appendix Section 1 task.
  2. A code cell that calls `fetch_hop2.py` via %run (so the full script runs
     in the notebook's kernel) OR runs the inline cell equivalent.

Run this ONCE from the terminal (with the .venv active):
    python inject_hop2_cells.py

It is safe to re-run — it checks for the cell ID before inserting.
"""

import json
import sys
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "hpfrac.ipynb")

MARKDOWN_CELL_ID = "true2hop_phase1_header"
CODE_CELL_ID     = "true2hop_phase1_code"

MARKDOWN_SOURCE = [
    "---\n",
    "## TRUE 2-HOP ARCHITECTURE — Phase 1: Data Engineering & The $P_c^2$ Layer\n",
    "### Appendix Section 1: Fault-Tolerant Hop-2 Citation Extraction\n",
    "\n",
    "**Goal**: Fetch the $P_c^2$ layer — papers that cite the Hop-1 citing papers ($P_c^1$) — using the Semantic Scholar batch API.\n",
    "\n",
    "**Why a new script is needed** (bugs in the previous Hop-2 cell `866bab97`):\n",
    "1. **Boundary-node error**: The old script filtered out any edge missing an `abstract` or `contexts`. This silently dropped frontier nodes — the most important Hop-2 edges. The new script saves *all* edges regardless of missing fields.\n",
    "2. **Checkpoint poisoning on 429**: The old script wrote IDs to the checkpoint log *before* verifying the API response. A 429 during a batch permanently marked those IDs as done. The new script only checkpoints after a successful Parquet write.\n",
    "3. **Data loss on exit**: The `hop2_checkpoints/` directory is **empty** after the prior run, confirming the final in-memory buffer was never flushed. The new script has an unconditional final flush.\n",
    "\n",
    "**Design**:\n",
    "- Uses the S2 batch endpoint (`POST /paper/batch`) with up to 100 IDs per request.\n",
    "- Exponential back-off on 429 (60 s → 120 s → 240 s → 300 s cap).\n",
    "- Flushes edge buffer to a Parquet chunk every `FLUSH_EVERY=500` papers AND unconditionally at loop exit.\n",
    "- Checkpoint file (`_processed_hop1_ids.txt`) is only written after the Parquet file is verified on disk.\n",
    "- Adds a final consolidation step: merges all chunks into `hop2_edges.parquet` with deduplication.\n",
    "\n",
    "**Output schema** (`hop2_edges.parquet`):\n",
    "```\n",
    "hop1_paper_id    str   — P_c^1 paper that was queried\n",
    "hop2_paper_id    str   — P_c^2 paper (the Hop-2 citation)\n",
    "hop2_title       str   — title (may be None)\n",
    "hop2_year        int   — publication year (may be None)\n",
    "hop2_abstract    str   — abstract text (None at frontier nodes is expected)\n",
    "hop2_author_ids  str   — JSON-encoded list of S2 author ID strings\n",
    "citation_context str   — in-text citation sentence (None if not extracted)\n",
    "```\n",
]

CODE_SOURCE = [
    "# ==============================================================================\n",
    "# Phase 1: Run the fault-tolerant P_c^2 extraction pipeline.\n",
    "#\n",
    "# This cell executes fetch_hop2.py in the current kernel via %run.\n",
    "# The script will automatically resume from where it left off if re-run.\n",
    "#\n",
    "# OPTIONAL: Set your free Semantic Scholar API key to unlock 10x rate limits:\n",
    "#   1. Register at https://www.semanticscholar.org/product/api\n",
    "#   2. Open fetch_hop2.py and set S2_API_KEY = \"YOUR_KEY_HERE\"\n",
    "# ==============================================================================\n",
    "\n",
    "import os\n",
    "script_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'fetch_hop2.py')\n",
    "# If running the notebook from the same directory as fetch_hop2.py, use:\n",
    "script_path = 'fetch_hop2.py'\n",
    "\n",
    "# Verify the script exists before running\n",
    "if not os.path.exists(script_path):\n",
    "    raise FileNotFoundError(\n",
    "        f\"fetch_hop2.py not found at {os.path.abspath(script_path)}. \"\n",
    "        \"Make sure it lives in the same directory as this notebook.\"\n",
    "    )\n",
    "\n",
    "# Run it!  The %run magic executes the file in the current kernel namespace.\n",
    "# All print/tqdm output will appear inline below this cell.\n",
    "%run fetch_hop2.py\n",
    "\n",
    "# After this cell completes, verify the output:\n",
    "import pandas as pd\n",
    "if os.path.exists('hop2_edges.parquet'):\n",
    "    _df = pd.read_parquet('hop2_edges.parquet', engine='fastparquet')\n",
    "    print(f\"\\n✅ hop2_edges.parquet loaded successfully.\")\n",
    "    print(f\"   Shape          : {_df.shape}\")\n",
    "    print(f\"   Unique P_c^2   : {_df['hop2_paper_id'].nunique():,}\")\n",
    "    print(f\"   Unique P_c^1   : {_df['hop1_paper_id'].nunique():,}\")\n",
    "    print(f\"   Columns        : {list(_df.columns)}\")\n",
    "else:\n",
    "    print(\"\\n⚠️  hop2_edges.parquet not found. Check the output above for errors.\")\n",
]

def already_has_cell(cells: list, cell_id: str) -> bool:
    return any(c.get("id") == cell_id for c in cells)

def make_markdown_cell(cell_id: str, source: list) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }

def make_code_cell(cell_id: str, source: list) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def main():
    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Guard: only insert if not already present
    if already_has_cell(cells, MARKDOWN_CELL_ID) or already_has_cell(cells, CODE_CELL_ID):
        print("Cells already present in the notebook — nothing to do.")
        sys.exit(0)

    md_cell   = make_markdown_cell(MARKDOWN_CELL_ID, MARKDOWN_SOURCE)
    code_cell = make_code_cell(CODE_CELL_ID, CODE_SOURCE)

    cells.append(md_cell)
    cells.append(code_cell)

    print(f"Appended 2 new cells (IDs: {MARKDOWN_CELL_ID}, {CODE_CELL_ID})")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("✅  Notebook updated successfully.")
    print("    Open hpfrac.ipynb and scroll to the bottom to see the new cells.")

if __name__ == "__main__":
    main()
