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


class EpochTimeout(Exception):
    pass


class Searcher:
    """Read path over an Indexer: snapshot epoch, visible mask, hybrid candidates. No locks held."""

    def __init__(self, indexer, embedder):
        self.ix, self.emb = indexer, embedder

    def search(
        self,
        query: str,
        k: int = 10,
        principal: str | None = None,
        min_epoch: int | None = None,
        at_epoch: int | None = None,
        timeout: float = 5.0,
    ) -> tuple[int, list[dict], dict]:
        from liveindex.index.visibility import visible_mask

        if min_epoch is not None and not self.ix.wait_for(min_epoch, timeout):
            raise EpochTimeout(f"index_epoch {self.ix.index_epoch} < min_epoch {min_epoch}")
        view = self.ix.view  # one read: epoch and array bounds are consistent
        e = view.epoch if at_epoch is None else min(at_epoch, view.epoch)
        if principal is None:
            allowed = np.ones(view.n_docs, dtype=bool)
        else:
            allowed = self.ix.acl.allowed_docs(principal, e, view.n_docs)
        mask = visible_mask(view.born, view.dead, view.doc_ix, view.n_chunks, e, allowed)
        qv = self.emb.encode_queries([query])[0]
        fused, meta = hybrid_search(self.ix.bm25, self.ix.hnsw, query, qv, mask, k)
        hits = [
            {
                "doc_id": self.ix.chunk_doc[c],
                "version": self.ix.chunk_version[c],
                "chunk_text": self.ix.chunk_text[c],
                "score": s,
                "epoch": e,
            }
            for c, s in fused
        ]
        meta["epoch"] = e
        return e, hits, meta
