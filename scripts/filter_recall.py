"""Recall of filtered HNSW vs exact search at several selectivities (prompt 1.4 report)."""
from __future__ import annotations

import json

import numpy as np

from liveindex.embed import Embedder
from liveindex.eval import beir_runner
from liveindex.eval.run import write_result
from liveindex.eval.vector_only import build_chunks
from liveindex.index.hnsw import HNSWIndex


def main() -> None:
    corpus, queries, qrels = beir_runner.load("scifact")
    emb = Embedder()
    texts, _owners, _ = build_chunks(corpus, emb)
    vecs = emb.encode(texts)
    qids = [q for q in qrels if q in queries][:300]
    qv = emb.encode_queries([queries[q] for q in qids])
    h = HNSWIndex(capacity=len(vecs))
    h.add(vecs)
    rng = np.random.default_rng(0)
    rows = []
    for sel in (0.5, 0.1, 0.02):
        r_graph, r_default, used_exact = [], [], 0
        for i in range(len(qids)):
            mask = rng.random(len(vecs)) < sel
            want = {c for c, _ in h.exact(qv[i], mask, 10)}
            g, _ = h.knn(qv[i], 10, mask, fallback_fraction=0.0)  # graph only
            d, ex = h.knn(qv[i], 10, mask)  # default: exact below 5%
            used_exact += ex
            r_graph.append(len({c for c, _ in g} & want) / max(len(want), 1))
            r_default.append(len({c for c, _ in d} & want) / max(len(want), 1))
        rows.append({
            "selectivity": sel, "visible_fraction": sel, "queries": len(qids),
            "recall_filtered_hnsw_graph_only": round(float(np.mean(r_graph)), 4),
            "recall_default_with_fallback": round(float(np.mean(r_default)), 4),
            "fallback_used_fraction": round(used_exact / len(qids), 3),
        })
    out = {"dataset": "scifact", "chunks": len(vecs), "ef": h.ef, "rows": rows}
    print(write_result("filter_recall_scifact", out).name)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
