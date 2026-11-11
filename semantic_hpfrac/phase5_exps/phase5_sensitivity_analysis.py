"""
Phase 5: sensitivity of the rankings to the hand-set parameters.

Sweeps intent-weight policies x lambda x delta. For EVERY grid point the metric
is recomputed from the graph with core.metrics (a decay rate applied per edge,
not a constant multiplier on final scores, which could never change a ranking).

Reported per grid point, against the reference config (core.metrics.FULL, or
FULL with --lam):
  - Kendall tau-b and top-100 overlap of the Hop-0 paper ranking
  - Kendall tau-b of the author ranking (thin data, see printed warnings)

Report whatever comes out. If rankings move a lot, that is a finding too; there
is no pass threshold. Once Phase 3/4 ground truth exists, add its AUC per grid
point (plan item 9) so stability can be weighed against accuracy.

Run:  python3 phase5_exps/phase5_sensitivity_analysis.py [--lam X] [--predictions FILE]
Output: reports/phase5_sensitivity_analysis.csv (long) and
        reports/phase5_sensitivity_matrix.csv (paper tau, policy x lambda, at the reference delta)
"""

import argparse
import os
from dataclasses import replace

import pandas as pd

from phase5_data import BANNER, DEFAULT_PREDICTIONS, REPORTS, describe, load_phase5_graph, rank_agreement
from core.metrics import FULL, author_metric, paper_scores

# (method, result, background); all keep method >= result >= background, plus all-equal
POLICIES = {
    "equal (1.0,1.0,1.0)": (1.0, 1.0, 1.0),
    "mild (1.2,1.1,1.0)": (1.2, 1.1, 1.0),
    "paper (1.5,1.2,1.0)": (1.5, 1.2, 1.0),
    "bg-penalty (1.5,1.2,0.5)": (1.5, 1.2, 0.5),
    "method-focus (2.0,1.2,0.5)": (2.0, 1.2, 0.5),
    "extreme (3.0,1.5,0.5)": (3.0, 1.5, 0.5),
}
LAMBDAS = [0.0, 0.05, 0.1, 0.18, 0.3]
DELTAS = [0.05, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None, help="reference lambda (default: core.metrics FULL.lam)")
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    print(BANNER, "\n")
    edges, authorship, info = load_phase5_graph(args.predictions)
    print(describe(info))

    ref = FULL if args.lam is None else replace(FULL, lam=args.lam)
    lambdas = sorted(set(LAMBDAS + [ref.lam]))
    print(f"Reference: lambda={ref.lam}, delta={ref.delta}, weights={ref.intent_weights}")
    print(f"Grid: {len(POLICIES)} policies x {len(lambdas)} lambdas x {len(DELTAS)} deltas\n")

    ref_paper = paper_scores(edges, ref)
    ref_author = author_metric(authorship, ref_paper, ref)

    rows = []
    for pol, (m, r, b) in POLICIES.items():
        for lam in lambdas:
            for delta in DELTAS:
                cfg = replace(ref, intent_weights={"method": m, "result": r, "background": b},
                              lam=lam, delta=delta)
                ps = paper_scores(edges, cfg)
                tau_p, top = rank_agreement(ps, ref_paper, args.top_k)
                tau_a, _ = rank_agreement(author_metric(authorship, ps, cfg), ref_author, args.top_k)
                rows.append({"policy": pol, "w_method": m, "w_result": r, "w_background": b,
                             "lambda": lam, "half_life_years": round(0.6931 / lam, 2) if lam else None,
                             "delta": delta,
                             "paper_tau_vs_ref": round(tau_p, 4),
                             f"paper_top{args.top_k}_overlap_vs_ref": round(top, 3),
                             "author_tau_vs_ref": round(tau_a, 4),
                             "paper_mean_score": round(float(ps.reindex(ref_paper.index, fill_value=0).mean()), 4)})

    out = pd.DataFrame(rows)
    matrix = out[out["delta"] == ref.delta].pivot(index="lambda", columns="policy", values="paper_tau_vs_ref")
    matrix = matrix[list(POLICIES)]
    os.makedirs(REPORTS, exist_ok=True)
    out.to_csv(os.path.join(REPORTS, "phase5_sensitivity_analysis.csv"), index=False)
    matrix.to_csv(os.path.join(REPORTS, "phase5_sensitivity_matrix.csv"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Paper-ranking Kendall tau vs reference (delta = {ref.delta}):")
        print(matrix.to_string())
    print(f"\nAcross all {len(out)} grid points: paper tau min {out['paper_tau_vs_ref'].min():.4f}, "
          f"median {out['paper_tau_vs_ref'].median():.4f}")
    print(f"Saved reports/phase5_sensitivity_analysis.csv and reports/phase5_sensitivity_matrix.csv in {REPORTS}")


if __name__ == "__main__":
    main()
