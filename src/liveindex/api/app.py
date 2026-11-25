"""In-memory FastAPI app (no epochs yet; epoch is a placeholder 0 until prompt 1.3)."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI

from liveindex.api.models import Hit, SearchRequest, SearchResponse, UpsertRequest
from liveindex.chunking import chunk_text
from liveindex.embed import Embedder
from liveindex.index.bm25 import BM25Index
from liveindex.index.hnsw import HNSWIndex
from liveindex.search import hybrid_search


def create_app(embedder: Embedder | None = None) -> FastAPI:
    app = FastAPI(title="LiveIndex")
    emb = embedder or Embedder()
    bm25, hnsw = BM25Index(), HNSWIndex()
    chunks: list[tuple[str, int, str]] = []

    @app.post("/v1/docs")
    def upsert(req: UpsertRequest) -> dict:
        parts = chunk_text(req.text, emb.count_tokens)
        bm25.add_chunks(parts)
        hnsw.add(emb.encode(parts))
        chunks.extend((req.doc_id, 1, p) for p in parts)
        return {"epoch": 0, "chunks": len(parts)}

    @app.post("/v1/search")
    def search(req: SearchRequest) -> SearchResponse:
        mask = np.ones(len(chunks), dtype=bool)
        qv = emb.encode_queries([req.query])[0]
        fused, _ = hybrid_search(bm25, hnsw, req.query, qv, mask, req.k)
        hits = [
            Hit(doc_id=chunks[c][0], version=chunks[c][1], chunk_text=chunks[c][2], score=s, epoch=0)
            for c, s in fused
        ]
        return SearchResponse(epoch=0, hits=hits)

    return app
