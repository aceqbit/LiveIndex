"""hnswlib wrapper: add, mark_deleted, filtered knn, and exact fallback over a mask."""
from __future__ import annotations

import os

import hnswlib
import numpy as np

EXACT_FALLBACK_FRACTION = float(os.environ.get("EXACT_FALLBACK_FRACTION", "0.05"))


class HNSWIndex:
    """Inner-product index over normalised vectors; labels are dense chunk ids."""

    def __init__(self, dim: int = 384, capacity: int = 1024, ef: int = 200, M: int = 16):
        self.dim, self.ef = dim, ef
        self._idx = hnswlib.Index(space="ip", dim=dim)
        self._idx.init_index(max_elements=capacity, ef_construction=200, M=M)
        self._idx.set_ef(ef)
        self._cap = capacity
        self.vecs = np.zeros((capacity, dim), dtype=np.float32)
        self.n = 0

    def add(self, vectors: np.ndarray) -> None:
        m = len(vectors)
        if m == 0:
            return
        if self.n + m > self._cap:
            new_cap = max(self.n + m, 2 * self._cap)
            self._idx.resize_index(new_cap)
            grown = np.zeros((new_cap, self.dim), dtype=np.float32)
            grown[: self.n] = self.vecs[: self.n]
            self.vecs, self._cap = grown, new_cap
        self.vecs[self.n : self.n + m] = vectors
        self._idx.add_items(vectors, np.arange(self.n, self.n + m))
        self.n += m

    def mark_deleted(self, chunk_id: int) -> None:
        self._idx.mark_deleted(int(chunk_id))

    def exact(self, query: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        n = min(self.n, len(mask))
        ids = np.flatnonzero(mask[:n])
        if len(ids) == 0:
            return []
        sims = self.vecs[ids] @ query
        top = np.argsort(-sims, kind="stable")[:k]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def knn(
        self, query: np.ndarray, k: int, mask: np.ndarray, fallback_fraction: float | None = None
    ) -> tuple[list[tuple[int, float]], bool]:
        """Return (hits, used_exact). The mask is applied inside the graph search."""
        n = min(self.n, len(mask))
        if n == 0:
            return [], False
        frac = EXACT_FALLBACK_FRACTION if fallback_fraction is None else fallback_fraction
        visible = int(mask[:n].sum())
        if visible == 0:
            return [], False
        if visible < frac * n:
            return self.exact(query, mask, k), True
        self._idx.set_ef(max(self.ef, k))
        kk = min(k, visible)
        try:
            labels, dists = self._idx.knn_query(
                query[None, :], k=kk, filter=lambda i: i < n and bool(mask[i])
            )
        except RuntimeError:  # graph could not find k filtered neighbours
            return self.exact(query, mask, k), True
        return [(int(i), float(1.0 - d)) for i, d in zip(labels[0], dists[0], strict=True)], False
