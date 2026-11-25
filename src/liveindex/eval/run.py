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
    if a.retriever != "vector":
        raise SystemExit(f"unknown retriever {a.retriever}")
    qs = {q: queries[q] for q in qrels if q in queries}
    run = VectorRetriever(emb, vecs, owners).search(qs, 10)
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
