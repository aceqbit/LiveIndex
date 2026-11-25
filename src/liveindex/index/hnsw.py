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
        self.M = M
        self._deleted: set[int] = set()
        self._idx = self._new_index(capacity)
        self._cap = capacity
        self.vecs = np.zeros((capacity, dim), dtype=np.float32)
        self.n = 0

    def _new_index(self, capacity: int) -> hnswlib.Index:
        idx = hnswlib.Index(space="ip", dim=self.dim)
        idx.init_index(max_elements=capacity, ef_construction=200, M=self.M)
        idx.set_ef(self.ef)
        return idx

    def add(self, vectors: np.ndarray) -> None:
        m = len(vectors)
        if m == 0:
            return
        if self.n + m > self._cap:
            # copy-on-grow: readers keep using the old graph and vectors until the swap
            new_cap = max(self.n + m, 2 * self._cap)
            idx = self._new_index(new_cap)
            if self.n:
                idx.add_items(self.vecs[: self.n], np.arange(self.n))
            for d in self._deleted:
                idx.mark_deleted(d)
            grown = np.zeros((new_cap, self.dim), dtype=np.float32)
            grown[: self.n] = self.vecs[: self.n]
            grown[self.n : self.n + m] = vectors
            idx.add_items(vectors, np.arange(self.n, self.n + m))
            self.vecs, self._idx, self._cap = grown, idx, new_cap
            self.n += m
            return
        self.vecs[self.n : self.n + m] = vectors
        self._idx.add_items(vectors, np.arange(self.n, self.n + m))
        self.n += m

    def mark_deleted(self, chunk_id: int) -> None:
        self._deleted.add(int(chunk_id))
        self._idx.mark_deleted(int(chunk_id))

    def exact(self, query: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        vecs = self.vecs
        n = min(self.n, len(mask), len(vecs))
        ids = np.flatnonzero(mask[:n])
        if len(ids) == 0:
            return []
        sims = vecs[ids] @ query
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
        idx = self._idx
        idx.set_ef(max(self.ef, k))
        kk = min(k, visible)
        try:
            labels, dists = idx.knn_query(
                query[None, :], k=kk, filter=lambda i: i < n and bool(mask[i])
            )
        except RuntimeError:  # graph could not find k filtered neighbours
            return self.exact(query, mask, k), True
        return [(int(i), float(1.0 - d)) for i, d in zip(labels[0], dists[0], strict=True)], False
