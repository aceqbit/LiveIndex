"""Ingest SciFact into the WAL, then time a cold replay (prompt 1.3 report)."""
from __future__ import annotations

import json
import sys
import time

from liveindex.embed import Embedder
from liveindex.eval import beir_runner
from liveindex.eval.run import write_result
from liveindex.indexer import Indexer
from liveindex.wal.log import WriteLog


def main() -> None:
    corpus, _, _ = beir_runner.load("scifact")
    emb = Embedder()
    log = WriteLog()
    log.reset()
    t0 = time.perf_counter()
    for did, doc in corpus.items():
        log.upsert(did, beir_runner.doc_text(doc), allowed=["u"])
    ingest = time.perf_counter() - t0
    ix = Indexer(log, emb)
    t0 = time.perf_counter()
    ix.catch_up()  # first replay may include embedding time (cache is warm from eval runs)
    replay = time.perf_counter() - t0
    ix2 = Indexer(log, emb)
    t0 = time.perf_counter()
    ix2.catch_up()
    replay2 = time.perf_counter() - t0
    out = {
        "dataset": "scifact", "docs": len(corpus), "chunks": ix2._n, "index_epoch": ix2.index_epoch,
        "wal_ingest_seconds": round(ingest, 2), "replay_seconds_first": round(replay, 2),
        "replay_seconds_restart": round(replay2, 2),
    }
    print(write_result("replay_scifact", out).name, json.dumps(out))
    log.close()


if __name__ == "__main__":
    sys.exit(main())
