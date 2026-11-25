"""In-house BM25 (k1=1.2, b=0.75): own tokenizer, numpy postings, mask-restricted scoring."""
from __future__ import annotations

import math
import re

import numpy as np

K1 = 1.2
B = 0.75
STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"]
)
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOP]


class BM25Index:
    """Postings are stored per term as (chunk_ids, tfs); doc lengths are a growable array.

    Single writer: `add_chunks` appends. Readers call `score` with a mask whose length
    bounds the chunk ids they may see, so a concurrent append is never half-observed.
    """

    def __init__(self) -> None:
        self._post: dict[str, tuple[list[int], list[int]]] = {}
        self._np: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        self._lens = np.zeros(1024, dtype=np.int64)
        self.n = 0
        self._total_len = 0

    def add_chunks(self, texts: list[str], start_id: int | None = None) -> None:
        cid = self.n if start_id is None else start_id
        if cid != self.n:
            raise ValueError("chunk ids must be dense and increasing")
        need = self.n + len(texts)
        if need > len(self._lens):
            grown = np.zeros(max(need, 2 * len(self._lens)), dtype=np.int64)
            grown[: self.n] = self._lens[: self.n]
            self._lens = grown
        for text in texts:
            toks = tokenize(text)
            self._lens[cid] = len(toks)
            self._total_len += len(toks)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                ids, tfs = self._post.setdefault(t, ([], []))
                ids.append(cid)
                tfs.append(c)
            cid += 1
        self.n = cid

    def _arrays(self, term: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Postings as arrays; the cache entry records its length so a concurrent append is never lost."""
        p = self._post.get(term)
        if p is None:
            return None
        length = min(len(p[0]), len(p[1]))
        c = self._np.get(term)
        if c is None or c[0] != length:
            c = (
                length,
                np.asarray(p[0][:length], dtype=np.int64),
                np.asarray(p[1][:length], dtype=np.float64),
            )
            self._np[term] = c
        return c[1], c[2]

    def score(self, query: str, mask: np.ndarray, top_n: int = 100) -> list[tuple[int, float]]:
        """Top-n (chunk_id, score) among chunks where mask is True. Never scores masked-out chunks."""
        n = min(self.n, len(mask))
        if n == 0:
            return []
        n_docs = self.n
        avgdl = self._total_len / n_docs if n_docs else 1.0
        scores = np.zeros(n, dtype=np.float64)
        for term in set(tokenize(query)):
            arr = self._arrays(term)
            if arr is None:
                continue
            ids, tfs = arr
            keep = ids < n
            ids, tfs = ids[keep], tfs[keep]
            ok = mask[ids]
            ids, tfs = ids[ok], tfs[ok]
            if len(ids) == 0:
                continue
            df = int(keep.sum())
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            dl = self._lens[ids]
            scores[ids] += idf * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * dl / avgdl))
        cand = np.flatnonzero(scores > 0)
        if len(cand) == 0:
            return []
        order = cand[np.lexsort((cand, -scores[cand]))][:top_n]
        return [(int(i), float(scores[i])) for i in order]
