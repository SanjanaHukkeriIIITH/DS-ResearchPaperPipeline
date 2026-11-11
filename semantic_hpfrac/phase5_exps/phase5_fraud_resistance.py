"""
Phase 5: manipulation test. Fake citations are ADDED TO THE GRAPH and every
index is recomputed with core.metrics. No outcome is assumed or hardcoded.

Target: Hop-0 papers (sampled with a fixed seed). For each target and each
attack size k we add k fake citing papers and measure the rise of:
  citations          raw citation count of the paper (what the h-index counts)
  hp (2-hop)         standard 2-hop paper score h_p (STANDARD config)
  semantic (2-hop)   Semantic Sh_p (FULL config)
  semantic (1-hop)   weighted citation sum S(p) (FULL with hops=1)

Attacks
  padding        k fake papers cite the target as Background; nobody cites them.
  ring_bg        same, but the fake papers form rings of --ring-size and cite
                 each other (Background), so each fake citer has ring-size - 1
                 citations of its own. This is the attack that targets 2-hop metrics.
  ring_method    ring, but the attacker words the ring citations so the classifier
                 labels them Method: tests what intent weighting gives if fooled.
All fake citations are 1 year old (attackers cite immediately: worst case for decay).

Why padding alone proves little: in the 2-hop formula a citer only counts if it
is itself cited, so uncited fake papers add nothing to hp OR semantic hp. Report
the ring attacks next to it.

Author level: the h-index, hp-frac and Semantic hp-frac of an author need authors
with many papers. The current authorship data has 1 Hop-0 paper per author, so
this script reports paper level only. Rerun the same attacks on the Phase 3
author crawl for the author table.

Run:  python3 phase5_exps/phase5_fraud_resistance.py [--lam X] [--n-targets 20] [--ring-size 10]
Output: reports/phase5_fraud_simulation_results.csv   (per target, attack, k)
        reports/phase5_fraud_summary.csv              (mean rise per index)
        reports/phase5_fraud_resistance_curve.png
"""

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from phase5_data import BANNER, DEFAULT_PREDICTIONS, REPORTS, describe, load_phase5_graph, uniform_alpha
from core.metrics import FULL, STANDARD, paper_scores

ATTACKS = ["padding", "ring_bg", "ring_method"]
LEVELS = [0, 10, 25, 50, 100, 200]
INDICES = ["citations", "hp (2-hop)", "semantic (2-hop)", "semantic (1-hop)"]


def subgraph(edges, target):
    h1 = edges[(edges["hop"] == 1) & (edges["cited_id"] == target)]
    h2 = edges[(edges["hop"] == 2) & edges["cited_id"].isin(h1["citing_id"])]
    return pd.concat([h1, h2], ignore_index=True)


def inject(sub, target, attack, k, ring_size):
    if k == 0:
        return sub
    fakes = [f"fake_{attack}_{i}" for i in range(k)]
    rows = [{"citing_id": f, "cited_id": target, "hop": 1, "intent": "background", "delta_t": 1.0}
            for f in fakes]
    if attack != "padding":
        intent = "method" if attack == "ring_method" else "background"
        for start in range(0, k, ring_size):
            ring = fakes[start:start + ring_size]
            rows += [{"citing_id": a, "cited_id": b, "hop": 2, "intent": intent, "delta_t": 1.0}
                     for a in ring for b in ring if a != b]
    fake = pd.DataFrame(rows)
    out = pd.concat([sub, fake], ignore_index=True)
    # Hop-1 attention is uniform anyway; fake papers get uniform attention over their in-edges.
    is_h1 = out["hop"] == 1
    out.loc[is_h1, "alpha"] = uniform_alpha(out[is_h1])
    is_fake_cited = out["cited_id"].str.startswith("fake_")
    if is_fake_cited.any():
        out.loc[is_fake_cited, "alpha"] = uniform_alpha(out[is_fake_cited])
    return out


def measure(g, target, full, one_hop):
    def score(cfg):
        return float(paper_scores(g, cfg).get(target, 0.0))
    return {"citations": float(((g["hop"] == 1) & (g["cited_id"] == target)).sum()),
            "hp (2-hop)": score(STANDARD),
            "semantic (2-hop)": score(full),
            "semantic (1-hop)": score(one_hop)}


def plot(summary, path, ring_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"citations": "#2a78d6", "hp (2-hop)": "#eb6834",
              "semantic (2-hop)": "#1baf7a", "semantic (1-hop)": "#eda100"}
    markers = {"citations": "o", "hp (2-hop)": "s", "semantic (2-hop)": "D", "semantic (1-hop)": "^"}
    titles = {"padding": "Padding (uncited fakes)",
              "ring_bg": f"Citation ring, size {ring_size} (Background)",
              "ring_method": f"Citation ring, size {ring_size} (labelled Method)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, dpi=200)
    for ax, attack in zip(axes, ATTACKS):
        d = summary[summary["attack"] == attack]
        for idx in INDICES:
            s = d[d["index"] == idx]
            ax.plot(s["k"], s["mean_rise"], marker=markers[idx], markersize=6, linewidth=2,
                    color=colors[idx], label=idx)
        ax.set_title(titles[attack], fontsize=10)
        ax.set_xlabel("fake citations added")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean rise in score (symlog)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=len(INDICES))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--ring-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(BANNER, "\n")
    edges, _, info = load_phase5_graph(args.predictions)
    print(describe(info))
    full = FULL if args.lam is None else replace(FULL, lam=args.lam)
    one_hop = replace(full, hops=1)
    print(f"lambda = {full.lam}, intent weights = {full.intent_weights}, ring size = {args.ring_size}\n")

    hop0 = np.sort(edges.loc[edges["hop"] == 1, "cited_id"].unique())
    targets = np.random.default_rng(args.seed).choice(hop0, size=min(args.n_targets, len(hop0)), replace=False)

    rows = []
    for t in targets:
        sub = subgraph(edges, t)
        base = measure(sub, t, full, one_hop)
        for attack in ATTACKS:
            for k in LEVELS:
                after = base if k == 0 else measure(inject(sub, t, attack, k, args.ring_size), t, full, one_hop)
                for idx in INDICES:
                    rows.append({"target_paper": t, "attack": attack, "k": k, "index": idx,
                                 "before": base[idx], "after": after[idx], "rise": after[idx] - base[idx]})
    res = pd.DataFrame(rows)
    summary = (res.groupby(["attack", "k", "index"], sort=False)
               .agg(mean_before=("before", "mean"), mean_after=("after", "mean"), mean_rise=("rise", "mean"))
               .reset_index())
    summary["rise_per_fake_citation"] = np.where(summary["k"] > 0, summary["mean_rise"] / summary["k"].clip(lower=1), 0.0)

    os.makedirs(REPORTS, exist_ok=True)
    res.to_csv(os.path.join(REPORTS, "phase5_fraud_simulation_results.csv"), index=False)
    summary.round(4).to_csv(os.path.join(REPORTS, "phase5_fraud_summary.csv"), index=False)
    plot(summary, os.path.join(REPORTS, "phase5_fraud_resistance_curve.png"), args.ring_size)

    wide = summary.pivot_table(index=["attack", "k"], columns="index", values="mean_rise", sort=False)[INDICES]
    with pd.option_context("display.width", 200):
        print(f"Mean rise per index over {len(targets)} target papers:")
        print(wide.round(3).to_string())
    print(f"\nSaved results, summary and plot to {REPORTS}")


if __name__ == "__main__":
    main()
