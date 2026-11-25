from __future__ import annotations

from pydantic import BaseModel


class UpsertRequest(BaseModel):
    doc_id: str
    text: str
    tenant: str = "default"


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class Hit(BaseModel):
    doc_id: str
    version: int
    chunk_text: str
    score: float
    epoch: int


class SearchResponse(BaseModel):
    epoch: int
    hits: list[Hit]
