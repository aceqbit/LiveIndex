"""Reciprocal rank fusion and the hybrid search function. All filtering happens before top-k."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from liveindex.index.bm25 import BM25Index
from liveindex.index.hnsw import HNSWIndex

RRF_K = 60


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K, top: int = 50) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]


def hybrid_search(
    bm25: BM25Index,
    hnsw: HNSWIndex,
    query: str,
    qvec: np.ndarray,
    mask: np.ndarray,
    k: int,
    ef: int | None = None,
) -> tuple[list[tuple[int, float]], dict]:
    lex = bm25.score(query, mask, 100)
    vec, exact = hnsw.knn(qvec, 100, mask)
    fused = rrf([[c for c, _ in lex], [c for c, _ in vec]], top=50)
    meta = {"fallback_exact": exact, "visible_fraction": float(mask.mean()) if len(mask) else 0.0}
    return fused[:k], meta
