import math

from liveindex.eval.metrics import evaluate, ndcg_at_k, recall_at_k


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}) == 1.0


def test_ndcg_hand_computed():
    # rels a=1,c=1; ranking [b, a, c]: dcg = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    got = ndcg_at_k(["b", "a", "c"], {"a": 1, "c": 1})
    want = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(got - want) < 1e-12


def test_ndcg_graded_and_cutoff():
    got = ndcg_at_k(["x", "a"], {"a": 2}, k=1)
    assert got == 0.0
    assert abs(ndcg_at_k(["a"], {"a": 2}, k=1) - 1.0) < 1e-12


def test_recall():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, 10) == 2 / 3
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 2) == 0.5
    assert recall_at_k(["a"], {}, 10) == 0.0


def test_evaluate_skips_queries_without_relevant():
    m = evaluate({"q1": ["a"], "q2": ["z"]}, {"q1": {"a": 1}, "q2": {"z": 0}})
    assert m["queries"] == 1 and m["ndcg@10"] == 1.0 and m["recall@10"] == 1.0
