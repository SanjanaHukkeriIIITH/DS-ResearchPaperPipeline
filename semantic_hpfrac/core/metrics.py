"""
Shared bibliometric metrics: h-index, hm-index, hp-frac and Semantic hp-frac.

Every experiment (main results, ablations, manipulation tests, top-author
table) must compute its numbers through this module so the paper reports one
consistent definition.

Input: an EDGE TABLE (pandas DataFrame), one row per citation edge
    citing_id : str    paper that cites
    cited_id  : str    paper being cited
    hop       : int    1 = Hop-1 -> Hop-0 edge, 2 = Hop-2 -> Hop-1 edge
    intent    : str    "background" | "method" | "result"  (needed if use_intent)
    alpha     : float  TemporalGAT attention, softmax-normalised over the
                       in-edges of cited_id present in this table (needed if use_attention)
    delta_t   : float  citing year - cited year, in years (needed if use_decay)

and an AUTHORSHIP TABLE, one row per (Hop-0 paper, author)
    paper_id, author_id

Definitions (hops=2, as in the paper):
    w(c'->c) = intent_weight * exp(-|lambda| * delta_t) * attention_term
    S(c)     = sum of w over Hop-2 edges into Hop-1 paper c
    Sh_p     = largest h such that >= h Hop-1 papers citing p have S(c) >= h
    author   = H([Sh_p / a_p for the author's papers]),  a_p = number of authors
With every component switched off, Sh_p reduces to the standard h_p, and the
author value to the standard hp-frac.

hops=1 uses the weighted citation count of p itself: S(p) = sum of w over
Hop-1 edges into p.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

INTENTS = ("background", "method", "result")


@dataclass(frozen=True)
class SemanticConfig:
    intent_weights: dict = field(
        default_factory=lambda: {"background": 1.0, "method": 1.5, "result": 1.2}
    )
    lam: float = 0.18          # placeholder: replace with the lambda learned in the Phase 1 retrain
    delta: float = 0.1         # baseline credit inside the attention term
    use_intent: bool = True
    use_decay: bool = True
    use_attention: bool = True
    fractional: bool = True    # divide paper scores by co-author count
    hops: int = 2


STANDARD = SemanticConfig(use_intent=False, use_decay=False, use_attention=False)

# Named configurations for the metric ablation (Phase 5). Each removes one
# component from FULL, or adds one component to STANDARD.
FULL = SemanticConfig()
ABLATIONS = {
    "standard_hp_frac": STANDARD,
    "full": FULL,
    "minus_intent": replace(FULL, use_intent=False),
    "minus_decay": replace(FULL, use_decay=False),
    "minus_attention": replace(FULL, use_attention=False),
    "minus_fractional": replace(FULL, fractional=False),
    "one_hop": replace(FULL, hops=1),
    "only_intent": replace(STANDARD, use_intent=True),
    "only_decay": replace(STANDARD, use_decay=True),
    "only_attention": replace(STANDARD, use_attention=True),
}


def _require(edges, cols, why):
    missing = [c for c in cols if c not in edges.columns]
    if missing:
        raise ValueError(f"edge table needs columns {missing} for {why}")
    nulls = [c for c in cols if edges[c].isna().any()]
    if nulls:
        raise ValueError(f"edge table has missing values in {nulls}, needed for {why}")


def edge_weights(edges, cfg):
    """Semantic weight w for every row of `edges` (numpy array)."""
    w = np.ones(len(edges))
    if cfg.use_intent:
        _require(edges, ["intent"], "use_intent")
        unknown = set(edges["intent"].unique()) - set(cfg.intent_weights)
        if unknown:
            raise ValueError(f"no intent weight for {sorted(unknown)}")
        w *= edges["intent"].map(cfg.intent_weights).to_numpy(dtype=float)
    if cfg.use_decay:
        _require(edges, ["delta_t"], "use_decay")
        w *= np.exp(-abs(cfg.lam) * edges["delta_t"].clip(lower=0).to_numpy(dtype=float))
    if cfg.use_attention:
        _require(edges, ["alpha"], "use_attention")
        # alpha sums to 1 over each cited paper's in-edges, so on its own it
        # only redistributes a fixed budget. Rescale so uniform attention
        # gives 1 per edge; the term then multiplies the citation count
        # instead of capping it.
        in_deg = edges.groupby("cited_id")["citing_id"].transform("size").to_numpy()
        rel = edges["alpha"].to_numpy(dtype=float) * in_deg
        w *= (rel + cfg.delta) / (1 + cfg.delta)
    return w


def hirsch(values):
    """Largest integer h such that at least h values are >= h."""
    v = np.sort(np.asarray(list(values), dtype=float))[::-1]
    ranks = np.arange(1, len(v) + 1)
    ok = v >= ranks
    return int(ranks[ok].max()) if ok.any() else 0


def hm_index(citations, n_authors):
    """Schreiber's hm-index: h-index with ranks counted fractionally (1/a per paper)."""
    df = pd.DataFrame({"c": citations, "a": n_authors}).sort_values("c", ascending=False)
    r_eff = np.cumsum(1.0 / df["a"].to_numpy(dtype=float))
    ok = df["c"].to_numpy(dtype=float) >= r_eff
    return float(r_eff[ok].max()) if ok.any() else 0.0


def node_scores(edges, cfg, hop):
    """S(x) = sum of w over edges of the given hop that point into x."""
    e = edges[edges["hop"] == hop]
    return pd.Series(edge_weights(e, cfg), index=e.index).groupby(e["cited_id"]).sum()


def paper_scores(edges, cfg):
    """Unfractionalised score of every Hop-0 paper (Sh_p for hops=2, S(p) for hops=1)."""
    hop1 = edges[edges["hop"] == 1]
    if cfg.hops == 1:
        return node_scores(edges, cfg, hop=1)
    if cfg.hops != 2:
        raise ValueError("hops must be 1 or 2")
    s_c = node_scores(edges, cfg, hop=2)
    citer_scores = hop1["citing_id"].map(s_c).fillna(0.0)
    return citer_scores.groupby(hop1["cited_id"]).apply(hirsch).astype(float)


def author_metric(authorship, scores, cfg):
    """Author-level index: H over (fractionalised) paper scores. Returns Series by author_id."""
    n_auth = authorship.groupby("paper_id")["author_id"].transform("size")
    s = authorship["paper_id"].map(scores).fillna(0.0)
    if cfg.fractional:
        s = s / n_auth
    return s.groupby(authorship["author_id"]).apply(hirsch).astype(float)


def author_table(edges, authorship, cfg=FULL):
    """h, hm, standard hp-frac and Semantic hp-frac (under `cfg`) for every author."""
    citations = edges[edges["hop"] == 1].groupby("cited_id").size()
    a = authorship.copy()
    a["citations"] = a["paper_id"].map(citations).fillna(0)
    a["n_authors"] = a.groupby("paper_id")["author_id"].transform("size")
    by_author = a.groupby("author_id")
    return pd.DataFrame({
        "h_index": by_author["citations"].apply(hirsch).astype(float),
        "hm_index": by_author[["citations", "n_authors"]].apply(
            lambda g: hm_index(g["citations"], g["n_authors"])),
        "hp_frac": author_metric(authorship, paper_scores(edges, STANDARD), STANDARD),
        "semantic_hp_frac": author_metric(authorship, paper_scores(edges, cfg), cfg),
    })
