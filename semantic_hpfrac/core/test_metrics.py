"""Run with:  python -m unittest core.test_metrics   (from semantic_hpfrac/)"""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from core.metrics import (
    ABLATIONS, FULL, STANDARD, author_table, edge_weights, hirsch, hm_index, paper_scores,
)


def toy_graph():
    """Hop-0 paper P is cited by Hop-1 papers C1..C3.
    C1 has 3 Hop-2 citers (method), C2 has 2 (background), C3 has 1 (result)."""
    rows = [("C1", "P", 1), ("C2", "P", 1), ("C3", "P", 1)]
    rows += [(f"x{i}", "C1", 2) for i in range(3)]
    rows += [(f"y{i}", "C2", 2) for i in range(2)]
    rows += [("z0", "C3", 2)]
    e = pd.DataFrame(rows, columns=["citing_id", "cited_id", "hop"])
    e["intent"] = ["method"] * 3 + ["method"] * 3 + ["background"] * 2 + ["result"]
    e["delta_t"] = 0.0
    e["alpha"] = 1.0 / e.groupby("cited_id")["citing_id"].transform("size")
    return e


class TestMetrics(unittest.TestCase):
    def test_hirsch(self):
        self.assertEqual(hirsch([]), 0)
        self.assertEqual(hirsch([10, 8, 5, 4, 3]), 4)
        self.assertEqual(hirsch([2.5, 2.5, 0.4]), 2)

    def test_hm_index(self):
        # all single-author -> hm equals h
        self.assertEqual(hm_index([10, 8, 5, 4, 3], [1] * 5), 4.0)
        # two authors each -> effective ranks 0.5, 1, 1.5, ...
        self.assertEqual(hm_index([3, 3, 3, 3], [2] * 4), 2.0)

    def test_standard_is_plain_citation_counts(self):
        e = toy_graph()
        self.assertTrue(np.allclose(edge_weights(e, STANDARD), 1.0))
        # citer counts 3, 2, 1 -> h_p = 2
        self.assertEqual(paper_scores(e, STANDARD)["P"], 2.0)

    def test_uniform_attention_and_zero_age_change_nothing(self):
        e = toy_graph()
        only_attention = ABLATIONS["only_attention"]
        only_decay = ABLATIONS["only_decay"]
        self.assertTrue(np.allclose(edge_weights(e, only_attention), 1.0))
        self.assertTrue(np.allclose(edge_weights(e, only_decay), 1.0))

    def test_intent_weights(self):
        e = toy_graph()
        # S(C1)=3*1.5=4.5, S(C2)=2*1.0=2, S(C3)=1.2 -> Sh_p = 2
        self.assertEqual(paper_scores(e, ABLATIONS["only_intent"])["P"], 2.0)
        heavy = replace(ABLATIONS["only_intent"], intent_weights={"background": 1, "method": 3, "result": 3})
        # S = 9, 2, 3 -> Sh_p = 2 ; raise background too -> 9, 6, 3 -> 3
        self.assertEqual(paper_scores(e, heavy)["P"], 2.0)
        heavy2 = replace(heavy, intent_weights={"background": 3, "method": 3, "result": 3})
        self.assertEqual(paper_scores(e, heavy2)["P"], 3.0)

    def test_decay_reduces(self):
        e = toy_graph()
        e.loc[e["hop"] == 2, "delta_t"] = 20.0
        self.assertLess(paper_scores(e, ABLATIONS["only_decay"])["P"], paper_scores(e, STANDARD)["P"])

    def test_one_hop(self):
        e = toy_graph()
        self.assertAlmostEqual(paper_scores(e, replace(STANDARD, hops=1))["P"], 3.0)

    def test_missing_intent_raises(self):
        e = toy_graph()
        e.loc[e["hop"] == 2, "intent"] = None
        with self.assertRaises(ValueError):
            paper_scores(e, FULL)

    def test_author_table(self):
        e = toy_graph()
        auth = pd.DataFrame({"paper_id": ["P", "P"], "author_id": ["A", "B"]})
        t = author_table(e, auth, FULL)
        self.assertEqual(t.loc["A", "h_index"], 1.0)     # one paper, 3 citations
        self.assertEqual(t.loc["A", "hp_frac"], 1.0)     # h_p=2 / 2 authors = 1
        self.assertEqual(list(t.index), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
