# Phase 5: Metric Ablations and Manipulation Test (corrected plan)

This replaces the earlier Phase 5 brief. It matches the plan the team agreed (Phase 5 = plan items 16–18, with the item 9 sensitivity sweep) and the shared code on the `phase0-core-metrics` branch.

## 1. What Phase 5 has to answer

The CIKM reviewers raised three points that Phase 5 answers:

- **"The weights look arbitrary."** Show how much the author and paper rankings change when the intent weights, λ and δ change.
- **"Why 2 hops?"** Compare 1-hop with 2-hop. This script shows how much the rankings differ. Whether 2-hop is *better* is decided later, on the Phase 3/4 ground truth (award winners, Test-of-Time papers), not by correlation.
- **"Fraud-resistant is claimed but never tested."** Add fake citations to the real graph and measure how much each index rises. If Semantic hp-frac does not resist, the claim comes out of the title and abstract.

Phase 5 does **not** crawl Semantic Scholar and does **not** train the HGNN.

## 2. Ground rules

1. **Every number comes from `core/metrics.py`.** No script computes h-index, hp-frac or Semantic hp-frac itself. This is the fix for the "results inconsistent with method" rejection.
2. **Always recompute from the citation graph.** Never multiply finished scores by a constant: a constant multiplier cannot change anyone's rank, so it makes ablations and sweeps meaningless.
3. **Never write expected results into the code, the plots or the doc.** Report what comes out, including results that go against us. There is no "τ > 0.85 = proof" threshold.
4. **λ is a placeholder until the Phase 1 retrain.** Do not hardcode it. 0.085 was a hand-set value, not a learned one, and the 0.0995 in `training_logs_scaled.csv` comes from a 5-epoch debug run. Run every experiment over the λ grid and pass the final value with `--lam` when Phase 1 is done.
5. **No Phase 5 number goes into the paper until Phase 1 is done.** Today's outputs test the pipeline only.

## 3. The metric (as in `core/metrics.py`)

- Edge weight: w(c→p) = intent weight × exp(−|λ|·Δt) × attention term, where Δt = citing year − cited year.
- Intent weights: Method 1.5, Result 1.2, Background **1.0** (same as the paper). These are a policy choice for now; Phase 2 derives them from data.
- Attention term: the TemporalGAT attention α, rescaled so equal attention gives 1, plus a small baseline δ = 0.1. The earlier brief left attention out; it is part of the metric and has its own ablation.
- S(c) = sum of the weights of the Hop-2 citations into Hop-1 paper c.
- Sh_p = largest h such that at least h Hop-1 papers citing p have S(c) ≥ h.
- Author score = H( Sh_p / a_p over the author's papers ), a_p = number of co-authors.
- With intent, decay and attention all off, this gives exactly standard hp-frac.

Important consequence for the manipulation test: in the 2-hop formula a citing paper only helps if it is **itself cited**. Fake papers that nobody cites add nothing to standard hp-frac or to Semantic hp-frac, whatever their intent label. So a padding attack alone cannot show that the *semantic* part helps; the citation-ring attack is the real test.

## 4. Data

| What | Where | Notes |
|---|---|---|
| Citation graph (Hop-0, Hop-1, Hop-2) | `data/cleaned/cleaned/hop{0,1,2}_v2.csv` | Load only through `core/loaders.py`, which fixes the swapped Hop-0/Hop-1 column names. Do not use `hop1_final_dataset_rescued.csv` / `hop2_final_dataset.csv`. |
| Hop-1 intents | SciCite gold labels (in the loader output) | |
| Hop-2 intents and attention | `data/data_v2/edge_predictions_scaled.parquet` | Old April model: stand-in only. Replace with the Phase 1 predictions via `--predictions`. |
| Authors | `data/paper_authors_edges.csv` | |

Real sizes after loading (rows, not file lines; abstracts span several lines, so `wc -l` over-counts):

- 4,447 Hop-0 papers, 4,476 Hop-1 edges kept (215 dropped: no year).
- 81,888 of 131,004 Hop-2 edges kept (they need a year and an old-model prediction). The same edge set is used for every variant.
- 927 authors on only 192 Hop-0 papers, one paper each.

### Known limits of today's data

The scripts print these as a warning each time they run.

1. **99.5% of Hop-0 papers have exactly one Hop-1 citer.** SciCite records one citing paper per pair, and we never crawled the other citers of Hop-0 papers. So every Hop-0 paper's 2-hop score is 0, 1 or 2, and paper rankings are almost binary.
2. **The old model's attention is almost uniform** (within about ±4%), so the attention ablation cannot show anything until Phase 1.
3. **Each author has one Hop-0 paper,** so author-level h-index / hp-frac rankings are not meaningful.

**What this means:** the Phase 5 code is complete, but its paper-ready results need (a) the Phase 1 predictions and (b) the Phase 3 author crawl, which collects all papers of each chosen author plus their citers. When that data exists, it is loaded into the same edge/authorship tables and the scripts are rerun unchanged.

## 5. The three scripts (`semantic_hpfrac/phase5_exps/`)

Shared input: `phase5_data.py` builds the edge and authorship tables and the rank-agreement helper. Run everything from any folder, e.g. `python3 phase5_exps/phase5_metric_ablations.py`.

### A. `phase5_metric_ablations.py` (Ablation B, plan items 16–17)

Uses the ready-made configs in `core.metrics.ABLATIONS`:

- standard hp-frac and full Semantic hp-frac
- full minus one component: intent, decay, attention, co-author fractions
- standard plus one component: intent only, decay only, attention only
- 1-hop instead of 2-hop

For each variant: Kendall τ-b and top-100 overlap of the paper ranking against full and against standard, plus author-ranking τ against full. Output: `reports/phase5_metric_ablations.csv`.

### B. `phase5_sensitivity_analysis.py` (plan item 9)

Grid: 6 intent policies (all with Method ≥ Result ≥ Background, plus all-equal) × λ ∈ {0, 0.05, 0.1, 0.18, 0.3} × δ ∈ {0.05, 0.1, 0.2}. Each grid point is recomputed from the graph. Reports paper τ, top-100 overlap and author τ against the reference config. Outputs: `reports/phase5_sensitivity_analysis.csv` and `reports/phase5_sensitivity_matrix.csv`. When Phase 4 ground truth exists, add its AUC per grid point.

### C. `phase5_fraud_resistance.py` (plan item 18)

Fake citing papers are added to the graph around 20 randomly chosen Hop-0 papers (fixed seed), k ∈ {10, 25, 50, 100, 200}:

- **padding:** fake papers cite the target as Background; nobody cites them.
- **ring (Background):** fake papers form rings of 10 and cite each other, so each fake citer has 9 citations of its own.
- **ring (labelled Method):** the same ring, if the attacker fools the classifier into labelling the ring citations Method.

All fake citations are one year old (worst case for decay). Measured per target: raw citation count, standard h_p, Semantic Sh_p (2-hop) and the 1-hop semantic score. Outputs: `reports/phase5_fraud_simulation_results.csv`, `reports/phase5_fraud_summary.csv` and `reports/phase5_fraud_resistance_curve.png` (one panel per attack).

The author-level version (h-index, hm-index, hp-frac and Semantic hp-frac of real authors) runs the same attacks once the Phase 3 author data is in.

## 6. What the pipeline test shows today (not for the paper)

With the old predictions and λ = 0.18, to show the code works:

- **Padding:** citation count rises by k and the 1-hop semantic score by about 0.84 per fake citation. Standard hp and Semantic hp do not move at all, because of the 2-hop structure and not because of the semantic weights.
- **Background ring:** standard hp rises by about 8.5 and Semantic hp by about 6.5, a modest extra resistance from decay.
- **Ring labelled Method:** Semantic hp rises *more* than standard hp (about 10.5). Intent weighting only helps if the classifier is not fooled.
- **Sensitivity:** paper-ranking τ stays between 0.91 and 1.0 across the grid, but on today's almost-binary scores that says little.

If these patterns hold on the real data, the honest paper claim is "2-hop structure resists padding; semantic weighting adds limited resistance to citation rings and depends on the classifier". It is not "fraud-resistant".

## 7. Checklist

- Do not crawl or retrain in Phase 5.
- Use `core/metrics.py` for every number; recompute from the graph.
- Sweep λ; pass the final λ and the Phase 1 predictions when they exist.
- Rerun all three scripts on the Phase 3 author data for the author-level tables.
- Write up whatever the results are; drop "fraud-resistant" from the title if the ring attacks are not resisted.
