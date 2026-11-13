# Phase 2: Replacing the hand-set intent weights (handoff brief)

Owner: Phase 2 teammate. Time box: 2–3 days. This runs in parallel with Phase 1 (HGNN retrain), Phase 3/4 (top-author crawl and ground truth) and Phase 5 (ablations). Read `phase5_exps/PHASE5_PLAN.md` first: the same ground rules apply.

## 1. The problem

The paper sets the intent weights by hand: **Method 1.5, Result 1.2, Background 1.0**. The CIKM reviewers called them arbitrary. Phase 2 has to give the weights a basis in data, or show honestly that the data does not support them.

λ (the decay rate) is **not** part of Phase 2. It comes from the Phase 1 retrain. δ (attention baseline) stays at 0.1 and is covered by the Phase 5 sweep.

## 2. Ground rules

1. **Do not fit the weights to the Phase 4 ground truth** (award winners, Test-of-Time papers). Phase 4 has roughly 30–60 authors. Tuning on it overfits and leaves nothing to evaluate on. Use it only as a held-out check at the end (section 6).
2. **Fix the method before looking at the result.** Write down in this file which method is the main one and how its output becomes weights. Then run it. Do not switch methods because one gives a nicer ordering.
3. **Report what comes out**, including an ordering that goes against the paper (see section 4: the first look already does).
4. **Every score goes through `core/metrics.py`.** Weights are passed in as `SemanticConfig(intent_weights={...})`. Do not change the defaults in `metrics.py` until the team agrees on the final weights.
5. **Do not derive weights from the old model's predicted intents.** The April model labels 77% of Hop-2 edges Method; SciCite's gold labels are 27% Method. Its predictions are badly skewed. Use gold labels, or the Phase 1 predictions once they exist.

## 3. What the weights do in the metric (read before choosing a method)

From `core/metrics.py`:

- w(c→p) = intent_weight × exp(−|λ|·Δt) × attention term
- S(c) = sum of w over Hop-2 citations into Hop-1 paper c
- Sh_p = largest h such that at least h citers c of p have S(c) ≥ h

Two consequences:

- **Weights act on Hop-2 edges only.** Hop-1 intents never enter the 2-hop score (they only enter the 1-hop variant). So what matters is the intent with which *Hop-2 papers cite Hop-1 papers*.
- **The scale matters, not just the ratios.** S(c) is compared with an integer h, so multiplying all weights by 2 changes rankings. Fixing Background = 1.0 is a convention that keeps the metric equal to standard hp-frac when every citation is Background. Keep that convention and state it.

## 4. Data you already have (no crawling needed for the main methods)

| What | Where | Notes |
|---|---|---|
| SciCite with gold intent and `isKeyCitation` | `scicite_raw_data/scicite/{train,dev,test}.jsonl` | 8,243 / 916 / 1,861 rows. **One row per citation mention**, not per paper pair: dedupe on (`citingPaperId`, `citedPaperId`) before computing rates. |
| Hop-1 edges with gold intent | `core.loaders.load_cleaned_v2()` | 4,691 edges: background 2,224, method 1,677, result 790 |
| Hop-2 edges with citation context | `data/cleaned/cleaned/hop2_v2.csv` (column `citation_text`), `data/citation_contexts_scaled.parquet` | No intent labels; needs Phase 1 predictions |
| Old model predictions | `data/data_v2/edge_predictions_scaled.parquet` | Stand-in only, skewed (rule 5) |
| Shared edge + author tables | `phase5_exps/phase5_data.py` → `load_phase5_graph()` | Use this for any metric run |

### First look (mention-level, not deduped: redo properly)

Share of citations S2 flags as key citations (`isKeyCitation`), by gold intent:

| split | background | method | result |
|---|---|---|---|
| train | 0.530 | 0.324 | 0.303 |
| dev | 0.470 | 0.325 | 0.228 |
| test | 0.309 | 0.291 | 0.154 |

**This is the opposite of the paper's ordering**: Background citations are flagged as key *more* often than Method ones. Before treating this as a result, check:

- Does it hold after deduping to one row per paper pair?
- S2's key-citation flag is driven largely by how many times the cited paper is mentioned in the citing text. Background citations may simply be mentioned more often. Control for mention count if you can (the number of SciCite rows per pair is a rough proxy).
- Why does the train split differ so much from test? (Different sampling in SciCite? Check the SciCite paper.)

If the reversal survives these checks, that is the finding and it goes in the paper.

## 5. Candidate methods

Pick **one main method and one contrast** before running (rule 2). My suggestion is A as main, B as contrast; change it if section 4's checks show A is unusable, but write down why.

**A. Influence rate (main).** weight(intent) = P(key citation | intent) / P(key citation | background). Use gold labels, deduped pairs, all three splits pooled (report per split too). Give 95% bootstrap intervals (resample paper pairs, 1,000 draws). External signal, no crawl, no circularity with Phase 4. Weakness: S2's flag is partly a mention-count signal (above).

**B. Information content (contrast).** weight(intent) = −log P(intent) / −log P(background), with P from SciCite gold labels. Rarer intents get more credit. Needs no extra signal, but it measures rarity, not influence, and it will rank Result highest. State that plainly.

**C. Downstream impact (stretch, only if A and B are done by day 2).** Do citations of a given intent predict that the cited paper is later cited more? For example, regress a Hop-1 paper's later citation count on the counts of each intent among its earlier citations. Needs intents on Hop-2 edges, so it waits for Phase 1 predictions. Watch the time windows: citations counted as features must come before those counted as the outcome.

**Not a method: fitting to Phase 4.** See rule 1 and section 6.

## 6. Evaluation

1. **Weight table:** for each method, the three weights with 95% intervals. Output `phase2_exps/reports/phase2_intent_weights.csv`.
2. **Do rankings care?** Add each derived weight set to `POLICIES` in `phase5_exps/phase5_sensitivity_analysis.py` and rerun it. The grid currently only holds orderings with Method ≥ Result ≥ Background; add the derived ones whatever their order. Report paper and author Kendall τ and top-100 overlap against the hand-set weights.
3. **Held-out check on Phase 4** (only once Phase 3/4 has data, and only as a check): matched-pair accuracy of Semantic hp-frac under the hand-set, A and B weights. Do not go back and change the weights based on this.

How to read the outcome:

- Derived weights close to the hand-set ones → "the weights are supported by data".
- Derived weights different but rankings barely move (τ near 1) → "the metric is robust to the weights"; that also answers the reviewers.
- Derived weights different and rankings move → report it, use the derived weights, and the paper's intent-weighting story changes.

## 7. Deliverables

- `phase2_exps/derive_intent_weights.py`: reads SciCite, dedupes, computes A and B with bootstrap intervals, writes the CSV. Runnable from any folder (copy the path setup from `phase5_exps/phase5_data.py`).
- `phase2_exps/reports/phase2_intent_weights.csv`.
- Derived policies added to the Phase 5 sensitivity grid, plus its rerun outputs.
- One paragraph for the paper's method section (how the weights are set) and one for results. Plain claims only; no "proves".

### What the existing Phase 5 sensitivity run does and does not show

`phase5_exps/reports_new/` already has a sensitivity sweep (90 configs, paper τ min 0.906, median 0.972 against the hand-set weights). It is **not** the Phase 2 answer:

- It ran on today's graph, where 99.5% of Hop-0 papers have one Hop-1 citer, so paper scores are only 0, 1 or 2 (mean 0.53–0.60). Nearly everything is tied, so τ is high whatever the weights. Author τ (median 1.0) is uninformative because each author has one paper.
- It uses the old model's Hop-2 intents (77% Method) and the placeholder λ = 0.18.
- All six policies keep Method ≥ Result ≥ Background, so it never tests the ordering the SciCite data points to (section 4).
- Robust weights are not the same as grounded weights: "the weights don't matter" invites "then why weight by intent at all?"
- Wording: τ = 0.906 means about 95% of paper pairs keep their order, not "90.6% correlated".
- λ moves the ranking more than the weight policy does (down a column vs. across a row of the matrix). λ gets its basis from the Phase 1 retrain.

Use it as the template for step 2 of section 6, not as evidence.

## 8. Schedule

Phase 1 predictions are expected the day after this brief is handed over.

| When | Work | Depends on |
|---|---|---|
| Today | Write down the main/contrast choice in this file. Dedupe SciCite, redo the section 4 table, do the mention-count check. Write `derive_intent_weights.py` with bootstrap intervals and produce the weight table. Add the derived weights to the Phase 5 grid and check it runs (pipeline test only). Draft the method paragraph. Ask the Phase 1 owner whether the new predictions file keeps the same columns. | Nothing: SciCite gold labels only |
| Tomorrow | Rerun the sensitivity grid with `--predictions <new file> --lam <learned λ>`. Start method C if time allows. Draft the results paragraph. | Phase 1 predictions and λ |
| Later | Final sensitivity numbers and the held-out Phase 4 check, on the Phase 3 author data. Until then, scores stay 0/1/2 and high τ is not evidence of robustness. | Phase 3 crawl, Phase 4 matched pairs |

## 9. Who to ask

- Metric definitions, `core/metrics.py`, Phase 5 scripts: Sanjana.
- Phase 1 predictions file and final λ: Phase 1 owner. Ask early whether the new predictions keep the column names in `edge_predictions_scaled.parquet` (`citing_paper_id`, `cited_paper_id`, `predicted_intent`, `raw_importance`), so `--predictions` works unchanged.
- Phase 4 matched pairs: Phase 3/4 owner.
