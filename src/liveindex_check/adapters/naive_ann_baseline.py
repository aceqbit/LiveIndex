"""naive_ann_baseline: the way many RAG stacks are wired, built on SQLite + hnswlib.

(pgvector was not installable on this machine, see DECISIONS D3; this is NOT pgvector.)

Two real-world defaults, and nothing else made worse than usual:

1. ACL is a post-filter. The ANN index returns its top-k nearest chunks first, and chunks the
   principal may not read are dropped afterwards. With a small k, other principals' chunks can
   crowd out visible ones, and the ACL a row carries is whatever the sync job last wrote.
2. Deletes (and superseded versions, and ACL changes) are applied by an asynchronous worker that
   runs `DELETE_LAG_S` seconds (default 2) after the write was acknowledged, like a nightly or
   queue-driven sync job. Inserts are synchronous.

`--fixed` sets lag 0 and filters inside the ANN search (allowed and live chunks only), which a
correct design must pass.

The adapter has no notion of visibility, so `settle` returns immediately: an acknowledgement is
taken at face value, as a client of such a system would.
"""
from __future__ import annotations

import heapq
import itertools
import json
import os
import sqlite3
import threading
import time
from typing import Any

import hnswlib
import numpy as np

from liveindex_check.adapter import Ack, Hit, Result

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


class _Embedder:
    def __init__(self) -> None:
        self._m: Any = None
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            if self._m is None:
                from sentence_transformers import SentenceTransformer

                self._m = SentenceTransformer(MODEL, device="cpu")
            v = self._m.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
            self._cache[text] = v
        return v


class NaiveAnnBaseline:
    def __init__(self, lag_s: float | None = None, fixed: bool = False):
        self.fixed = fixed
        self.lag = 0.0 if fixed else (lag_s if lag_s is not None else float(os.environ.get("DELETE_LAG_S", "2")))
        self.emb = _Embedder()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._init_state()

    # ---- state ----
    def _init_state(self) -> None:
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute(
            "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_id TEXT, version INT, text TEXT, allowed TEXT)"
        )
        self.db.execute("CREATE INDEX by_doc ON chunks (doc_id)")
        self.cap = 256
        self.idx = hnswlib.Index(space="cosine", dim=DIM)
        self.idx.init_index(max_elements=self.cap, ef_construction=200, M=16)
        self.idx.set_ef(100)
        self.next_id = 0
        self.jobs: list[tuple[float, int, tuple]] = []
        self.seq = itertools.count()
        self.n = 0

    def reset(self) -> None:
        self.close()
        with self._lock:
            self._init_state()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="baseline-sync")
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(2)
            self._worker = None

    # ---- async sync worker ----
    def _schedule(self, job: tuple) -> None:
        if self.lag <= 0:
            self._apply(job)
        else:
            heapq.heappush(self.jobs, (time.monotonic() + self.lag, next(self.seq), job))

    def _run(self) -> None:
        stop = self._stop
        while not stop.is_set():
            with self._lock:
                while self.jobs and self.jobs[0][0] <= time.monotonic():
                    self._apply(heapq.heappop(self.jobs)[2])
            stop.wait(0.01)

    def _apply(self, job: tuple) -> None:
        if job[0] == "delete":
            for cid in job[1]:
                self.db.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
                try:
                    self.idx.mark_deleted(cid)
                except RuntimeError:
                    pass
        elif job[0] == "acl":
            self.db.execute("UPDATE chunks SET allowed=? WHERE doc_id=?", (json.dumps(sorted(job[2])), job[1]))

    # ---- adapter API ----
    def _chunk_ids(self, doc_id: str) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))]

    def upsert(self, doc_id: str, text: str, allowed: set[str], tenant: str) -> Ack:
        vec = self.emb.encode(text)
        with self._lock:
            old = self._chunk_ids(doc_id)
            version = 1 + max(
                [r[0] for r in self.db.execute("SELECT version FROM chunks WHERE doc_id=?", (doc_id,))] or [0]
            )
            if self.next_id >= self.cap:
                self.cap *= 2
                self.idx.resize_index(self.cap)
            cid = self.next_id
            self.next_id += 1
            self.db.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?)", (cid, doc_id, version, text, json.dumps(sorted(allowed)))
            )
            self.idx.add_items(vec[None, :], np.array([cid]))  # insert is synchronous
            if old:
                self._schedule(("delete", old))  # the superseded version goes away later
        return Ack(f"{doc_id}:{version}", None)

    def delete(self, doc_id: str) -> Ack:
        with self._lock:
            ids = self._chunk_ids(doc_id)
            self._schedule(("delete", ids))
        return Ack(f"del:{doc_id}", None)

    def set_acl(self, doc_id: str, allowed: set[str]) -> Ack:
        with self._lock:
            self._schedule(("acl", doc_id, set(allowed)))
        return Ack(f"acl:{doc_id}", None)

    def search(self, principal: str, query: str, k: int, min_ack: Ack | None) -> Result:
        qv = self.emb.encode(query)
        with self._lock:
            live = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if live == 0 or self.next_id == 0:
                return Result([], None)
            if self.fixed:
                rows = {
                    r[0]: r for r in self.db.execute("SELECT chunk_id, doc_id, version, text, allowed FROM chunks")
                }
                ok = {c for c, r in rows.items() if principal in json.loads(r[4])}
                if not ok:
                    return Result([], None)
                labels, _ = self.idx.knn_query(qv[None, :], k=min(k, len(ok)), filter=lambda i: i in ok)
                return Result([Hit(rows[i][1], rows[i][2], rows[i][3]) for i in labels[0] if i in rows], None)
            # naive: ANN top-k first ...
            try:
                labels, _ = self.idx.knn_query(qv[None, :], k=min(k, max(live, 1)))
            except RuntimeError:
                return Result([], None)
            hits: list[Hit] = []
            for cid in labels[0]:
                row = self.db.execute(
                    "SELECT doc_id, version, text, allowed FROM chunks WHERE chunk_id=?", (int(cid),)
                ).fetchone()
                # ... then post-filter by ACL
                if row is not None and principal in json.loads(row[3]):
                    hits.append(Hit(row[0], row[1], row[2]))
            return Result(hits, None)

    def settle(self, ack: Ack, timeout_s: float) -> bool:
        return True


def make(args) -> NaiveAnnBaseline:
    return NaiveAnnBaseline(fixed=bool(getattr(args, "fixed", False)))
