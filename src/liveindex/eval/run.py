"""Evaluate retrievers on BEIR: `python -m liveindex.eval.run --dataset scifact --retriever vector`."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from liveindex.embed import MODEL_NAME, Embedder
from liveindex.eval import beir_runner
from liveindex.eval.metrics import evaluate
from liveindex.eval.vector_only import VectorRetriever, build_chunks

RESULTS = Path(__file__).resolve().parents[3] / "results"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {"git_sha": git_sha(), "utc": utc_stamp(), **payload}
    path = RESULTS / f"{name}_{payload['utc']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def engine_run(kind, emb, texts, vecs, owners, qs, k=10):
    """BM25 or hybrid (BM25 + HNSW, RRF) through the engine's own index classes."""
    import numpy as np

    from liveindex.index.bm25 import BM25Index
    from liveindex.index.hnsw import HNSWIndex
    from liveindex.search import hybrid_search

    bm = BM25Index()
    bm.add_chunks(texts)
    hn = HNSWIndex(capacity=len(texts))
    hn.add(vecs)
    mask = np.ones(len(texts), dtype=bool)
    qids = list(qs)
    qv = emb.encode_queries([qs[q] for q in qids])
    run = {}
    for i, q in enumerate(qids):
        if kind == "bm25":
            ranked = [c for c, _ in bm.score(qs[q], mask, 100)]
        else:
            ranked = [c for c, _ in hybrid_search(bm, hn, qs[q], qv[i], mask, 100)[0]]
        docs: list[str] = []
        for c in ranked:
            if owners[c] not in docs:
                docs.append(owners[c])
            if len(docs) == k:
                break
        run[q] = docs
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--retriever", default="vector")
    a = ap.parse_args()
    corpus, queries, qrels = beir_runner.load(a.dataset)
    emb = Embedder()
    t0 = time.time()
    texts, owners, multi = build_chunks(corpus, emb)
    vecs = emb.encode(texts)
    qs = {q: queries[q] for q in qrels if q in queries}
    if a.retriever == "vector":
        run = VectorRetriever(emb, vecs, owners).search(qs, 10)
    elif a.retriever in ("bm25", "hybrid"):
        run = engine_run(a.retriever, emb, texts, vecs, owners, qs)
    else:
        raise SystemExit(f"unknown retriever {a.retriever}")
    m = evaluate(run, qrels, 10)
    p = write_result(
        f"beir_{a.retriever}_{a.dataset}",
        {
            "dataset": a.dataset, "retriever": a.retriever, "model": MODEL_NAME,
            "docs": len(corpus), "chunks": len(texts), "docs_split_into_multiple_chunks": multi,
            "queries_evaluated": m["queries"], "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"],
            "seconds": round(time.time() - t0, 1),
        },
    )
    print(p.name, json.dumps({k: m[k] for k in ("ndcg@10", "recall@10")}), f"chunks={len(texts)} docs={len(corpus)}")


if __name__ == "__main__":
    main()
