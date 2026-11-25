"""nDCG@k and recall@k with linear gains (matches BEIR / pytrec_eval ndcg_cut)."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    gains = [max(rels.get(d, 0), 0) for d in ranked[:k]]
    return dcg(gains) / dcg(ideal)


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def evaluate(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]], k: int = 10
) -> dict[str, float]:
    """Mean metrics over queries that have at least one relevant doc."""
    qs = [q for q in qrels if any(r > 0 for r in qrels[q].values())]
    if not qs:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, "queries": 0}
    nd = sum(ndcg_at_k(run.get(q, []), qrels[q], k) for q in qs) / len(qs)
    rc = sum(recall_at_k(run.get(q, []), qrels[q], k) for q in qs) / len(qs)
    return {f"ndcg@{k}": nd, f"recall@{k}": rc, "queries": len(qs)}
