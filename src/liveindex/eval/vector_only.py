"""Exact (brute-force) dense retriever over chunk embeddings; doc score = best chunk.

Replaces the "pgvector-only" retriever of the original plan (pgvector is unavailable, DECISIONS D3).
"""
from __future__ import annotations

import numpy as np

from liveindex.chunking import chunk_text
from liveindex.embed import Embedder
from liveindex.eval.beir_runner import doc_text


def build_chunks(corpus: dict, emb: Embedder, target: int = 300, overlap: float = 0.15):
    """Return (chunk_texts, chunk_doc_ids, n_docs_chunked)."""
    texts: list[str] = []
    owners: list[str] = []
    multi = 0
    for did, doc in corpus.items():
        cs = chunk_text(doc_text(doc), emb.count_tokens, target, overlap)
        if len(cs) > 1:
            multi += 1
        texts.extend(cs)
        owners.extend([did] * len(cs))
    return texts, owners, multi


class VectorRetriever:
    def __init__(self, emb: Embedder, chunk_vecs: np.ndarray, owners: list[str]):
        self.emb, self.vecs, self.owners = emb, chunk_vecs, np.array(owners)

    def search(self, queries: dict[str, str], k: int = 10) -> dict[str, list[str]]:
        qids = list(queries)
        qv = self.emb.encode_queries([queries[q] for q in qids])
        sims = qv @ self.vecs.T
        run: dict[str, list[str]] = {}
        for i, q in enumerate(qids):
            order = np.argsort(-sims[i])
            seen: list[str] = []
            for j in order:
                d = str(self.owners[j])
                if d not in seen:
                    seen.append(d)
                    if len(seen) == k:
                        break
            run[q] = seen
        return run
