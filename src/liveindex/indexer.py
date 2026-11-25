"""Single-writer indexer: applies WAL rows in epoch order and publishes immutable views."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from liveindex.acl import INF, AclState
from liveindex.chunking import chunk_text
from liveindex.index.bm25 import BM25Index
from liveindex.index.hnsw import HNSWIndex
from liveindex.wal.log import WalRow, WriteLog


@dataclass(frozen=True)
class View:
    """What a reader may use. Epoch and array bounds are published together."""

    epoch: int
    born: np.ndarray
    dead: np.ndarray
    doc_ix: np.ndarray
    n_chunks: int
    n_docs: int


class Indexer:
    def __init__(self, log: WriteLog, embedder, capacity: int = 1024):
        self.log, self.emb = log, embedder
        self.bm25 = BM25Index()
        self.hnsw = HNSWIndex(capacity=capacity)
        self.acl = AclState()
        self.chunk_text: list[str] = []
        self.chunk_doc: list[str] = []
        self.chunk_version: list[int] = []
        self.doc_index: dict[str, int] = {}
        self.live_chunks: dict[str, list[int]] = {}
        self._born = np.zeros(capacity, dtype=np.int64)
        self._dead = np.full(capacity, INF, dtype=np.int64)
        self._dix = np.zeros(capacity, dtype=np.int32)
        self._n = 0
        self.view = View(0, self._born, self._dead, self._dix, 0, 0)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def index_epoch(self) -> int:
        return self.view.epoch

    def _grow(self, need: int) -> None:
        """Copy-on-grow: readers holding an older view keep their own arrays."""
        if need <= len(self._born):
            return
        cap = max(need, 2 * len(self._born))
        for name, fill, dt in (("_born", 0, np.int64), ("_dead", INF, np.int64), ("_dix", 0, np.int32)):
            old = getattr(self, name)
            new = np.full(cap, fill, dtype=dt)
            new[: self._n] = old[: self._n]
            setattr(self, name, new)

    def _apply_upsert(self, row: WalRow) -> None:
        ix = self.doc_index.setdefault(row.doc_id, len(self.doc_index))
        for c in self.live_chunks.get(row.doc_id, []):
            self._dead[c] = row.epoch
        text = row.payload["text"]
        parts = chunk_text(text, self.emb.count_tokens) or [text or " "]
        vecs = self.emb.encode(parts)
        start = self._n
        self._grow(start + len(parts))
        end = start + len(parts)
        self._born[start:end] = row.epoch
        self._dead[start:end] = INF
        self._dix[start:end] = ix
        self.chunk_text.extend(parts)
        self.chunk_doc.extend([row.doc_id] * len(parts))
        self.chunk_version.extend([row.version or 0] * len(parts))
        self.bm25.add_chunks(parts)
        self.hnsw.add(vecs)
        self._n = end
        self.live_chunks[row.doc_id] = list(range(start, end))

    def _apply_delete(self, row: WalRow) -> None:
        for c in self.live_chunks.pop(row.doc_id, []):
            self._dead[c] = row.epoch  # first set dead_epoch, then tombstone in the graph
            self.hnsw.mark_deleted(c)
        ix = self.doc_index.get(row.doc_id)
        if ix is not None:
            self.acl.revoke_all(ix, row.epoch)

    def _apply_row(self, row: WalRow) -> None:
        if row.op == "upsert":
            self._apply_upsert(row)
        elif row.op == "delete":
            self._apply_delete(row)
        elif "group" in row.payload:
            f = self.acl.group_add if row.op == "acl_grant" else self.acl.group_remove
            f(row.payload["group"], row.payload["member"], row.epoch)
        else:
            ix = self.doc_index.setdefault(row.doc_id, len(self.doc_index))
            g = self.acl.grant if row.op == "acl_grant" else self.acl.revoke
            g(ix, row.payload["principal"], row.epoch)

    def apply_epoch(self, epoch: int, rows: list[WalRow]) -> None:
        for row in rows:
            self._apply_row(row)
        self._publish(epoch)

    def _publish(self, epoch: int) -> None:
        view = View(epoch, self._born, self._dead, self._dix, self._n, len(self.doc_index))
        with self._cv:
            self.view = view  # publish: arrays and epoch become visible together
            self._cv.notify_all()

    def catch_up(self) -> int:
        # Read the committed epoch first: every epoch <= it has all its WAL rows committed already.
        committed = self.log.committed_epoch()
        rows = self.log.read_after(self.index_epoch)
        by_epoch: dict[int, list[WalRow]] = {}
        for r in rows:
            by_epoch.setdefault(r.epoch, []).append(r)
        for e in sorted(by_epoch):
            if e <= committed:
                self.apply_epoch(e, by_epoch[e])
        if committed > self.index_epoch:
            self._publish(committed)  # epochs that wrote no WAL rows (no-op ACL changes)
        return len(by_epoch)

    def wake(self, _epoch: int = 0) -> None:
        self._wake.set()

    def start(self) -> None:
        self.catch_up()  # recovery: replay the log from epoch 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="indexer")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            try:
                self.catch_up()
            except Exception:
                logging.getLogger(__name__).exception("indexer apply failed")
                time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(2)

    def wait_for(self, epoch: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self.view.epoch < epoch:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True
