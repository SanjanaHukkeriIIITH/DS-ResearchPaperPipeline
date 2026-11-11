"""
Phase 5, Ablation B: metric component ablation.

Every variant is a config from core.metrics.ABLATIONS (one component removed from
the full metric, or one component added to standard hp-frac, plus 1-hop vs 2-hop).
Each one is recomputed from the citation graph; nothing is scaled after the fact.

For every variant we report, against the FULL metric and against STANDARD hp-frac:
  - Kendall tau-b of the Hop-0 paper ranking (main result: 4.5k papers)
  - overlap of the top-100 papers
  - Kendall tau-b of the author ranking (secondary: authorship data is thin,
    see the printed counts; redo on the Phase 3 author crawl)

Agreement only says how much a component CHANGES the ranking, not whether the
change is better. "Is 2-hop better than 1-hop?" is answered by the Phase 3/4
ground-truth tasks, not here.

Run from anywhere:
    python3 phase5_exps/phase5_metric_ablations.py [--lam 0.18] [--predictions FILE]
Output: phase5_exps/reports/phase5_metric_ablations.csv
"""

import argparse
import os
from dataclasses import replace

import pandas as pd

from phase5_data import BANNER, DEFAULT_PREDICTIONS, REPORTS, describe, load_phase5_graph, rank_agreement
from core.metrics import ABLATIONS, FULL, STANDARD, author_metric, paper_scores

DESCRIPTIONS = {
    "standard_hp_frac": "Standard hp-frac (all components off)",
    "full": "Full Semantic hp-frac",
    "minus_intent": "Full minus intent weights",
    "minus_decay": "Full minus temporal decay",
    "minus_attention": "Full minus attention",
    "minus_fractional": "Full minus co-author fractions (author level only)",
    "one_hop": "Full, but 1-hop instead of 2-hop",
    "only_intent": "Standard plus intent weights",
    "only_decay": "Standard plus temporal decay",
    "only_attention": "Standard plus attention",
}


def with_lam(cfg, lam):
    return cfg if lam is None else replace(cfg, lam=lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None, help="decay rate (default: core.metrics FULL.lam)")
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    print(BANNER, "\n")
    edges, authorship, info = load_phase5_graph(args.predictions)
    print(describe(info))
    full = with_lam(FULL, args.lam)
    print(f"lambda = {full.lam}  intent weights = {full.intent_weights}  delta = {full.delta}\n")

    configs = {name: with_lam(cfg, args.lam) for name, cfg in ABLATIONS.items()}
    paper = {name: paper_scores(edges, cfg) for name, cfg in configs.items()}
    author = {name: author_metric(authorship, paper[name], cfg) for name, cfg in configs.items()}

    rows = []
    for name, cfg in configs.items():
        tau_p_full, top_full = rank_agreement(paper[name], paper["full"], args.top_k)
        tau_p_std, top_std = rank_agreement(paper[name], paper["standard_hp_frac"], args.top_k)
        tau_a_full, _ = rank_agreement(author[name], author["full"], args.top_k)
        rows.append({
            "variant": name,
            "description": DESCRIPTIONS.get(name, name),
            "hops": cfg.hops,
            "intent": cfg.use_intent,
            "decay": cfg.use_decay,
            "attention": cfg.use_attention,
            "fractional": cfg.fractional,
            "lambda": cfg.lam if cfg.use_decay else None,
            "paper_tau_vs_full": round(tau_p_full, 4),
            f"paper_top{args.top_k}_overlap_vs_full": round(top_full, 3),
            "paper_tau_vs_standard": round(tau_p_std, 4),
            f"paper_top{args.top_k}_overlap_vs_standard": round(top_std, 3),
            "author_tau_vs_full": round(tau_a_full, 4),
            "paper_mean_score": round(float(paper[name].reindex(paper["full"].index, fill_value=0).mean()), 4),
            "author_mean_score": round(float(author[name].mean()), 4),
        })

    out = pd.DataFrame(rows)
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "phase5_metric_ablations.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.drop(columns=["description"]).to_string(index=False))
    print(f"\nSaved {path}")
    print("Note: paper-level scores ignore co-author fractions, so 'minus_fractional' is 1.0 there by construction.")


if __name__ == "__main__":
    main()
