"""FastAPI app: writes commit to Postgres and return an epoch; reads use the indexer's snapshot."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException

from liveindex.api.models import AclRequest, Hit, SearchRequest, SearchResponse, UpsertRequest
from liveindex.indexer import Indexer
from liveindex.search import EpochTimeout, Searcher
from liveindex.wal.log import WriteLog


def principal_from(x_test_principal: str | None) -> str:
    """Test-only principal header, honoured only when ALLOW_TEST_PRINCIPAL=1 (replaced by tokens in 1.4)."""
    if os.environ.get("ALLOW_TEST_PRINCIPAL") == "1" and x_test_principal:
        return x_test_principal
    raise HTTPException(401, "no verified principal")


def create_app(embedder=None, log: WriteLog | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        emb = embedder
        if emb is None:
            from liveindex.embed import Embedder

            emb = Embedder()
        wl = log or WriteLog()
        ix = Indexer(wl, emb)
        wl.on_commit = ix.wake
        ix.start()
        state.update(log=wl, ix=ix, searcher=Searcher(ix, emb))
        yield
        ix.stop()

    app = FastAPI(title="LiveIndex", lifespan=lifespan)

    @app.post("/v1/docs")
    def upsert(req: UpsertRequest) -> dict:
        return {"epoch": state["log"].upsert(req.doc_id, req.text, req.tenant, req.allowed)}

    @app.delete("/v1/docs/{doc_id}")
    def delete(doc_id: str) -> dict:
        try:
            return {"epoch": state["log"].delete(doc_id)}
        except KeyError:
            raise HTTPException(404, "no such document") from None

    @app.put("/v1/docs/{doc_id}/acl")
    def put_acl(doc_id: str, req: AclRequest) -> dict:
        try:
            return {"epoch": state["log"].set_acl(doc_id, req.allowed)}
        except KeyError:
            raise HTTPException(404, "no such document") from None

    @app.get("/v1/epoch")
    def epoch() -> dict:
        return {"committed": state["log"].committed_epoch(), "index_epoch": state["ix"].index_epoch}

    @app.post("/v1/search")
    def search(req: SearchRequest, x_test_principal: str | None = Header(default=None)) -> SearchResponse:
        principal = principal_from(x_test_principal)
        try:
            e, hits, _ = state["searcher"].search(req.query, req.k, principal, req.min_epoch)
        except EpochTimeout:
            raise HTTPException(504, "index did not reach min_epoch in time") from None
        return SearchResponse(epoch=e, hits=[Hit(**h) for h in hits])

    @app.post("/v1/admin/reset")
    def reset() -> dict:
        if os.environ.get("ALLOW_TEST_PRINCIPAL") != "1":
            raise HTTPException(403, "disabled")
        ix = state["ix"]
        ix.stop()
        state["log"].reset()
        new = Indexer(state["log"], ix.emb)
        state["log"].on_commit = new.wake
        new.start()
        state.update(ix=new, searcher=Searcher(new, ix.emb))
        return {"epoch": 0}

    return app
