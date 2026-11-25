# DECISIONS

## D1: PostgreSQL 18 instead of 16
The machine already runs PostgreSQL 18 on localhost:5432. We use it as-is (database `liveindex` only). The schema in CLAUDE.md section 5 uses nothing version-specific.

## D2: no Docker, no Compose, no `make`
Native Windows 11 with PowerShell, no WSL, no Docker. `just` replaces `make` (justfile, PowerShell shell). Docker, Compose, Terraform, cloud, Prometheus, Grafana and OTel Collector instructions are out of scope. Spans go to the console and `results/traces.jsonl`.

## D3: pgvector not available; baseline is `naive_ann_baseline`
The PG18 server has no `vector` extension, and the Windows install from the pgvector README needs the MSVC C++ build tools (`vswhere` finds no VC tools component; `nmake` and `cl` are absent). Rather than install a multi-GB toolchain, the baseline will be built on SQLite plus hnswlib with the same two flaws (ACL post-filter after ANN top-k, asynchronous delete). It is named `naive_ann_baseline`. We never claim pgvector. The `chunks_baseline` pgvector table is therefore not created in `001_init.sql`.

## D4: `chroma-hnswlib==0.7.5` instead of `hnswlib`
`hnswlib` has no Windows wheel for Python 3.12 and needs MSVC to build. `chroma-hnswlib` 0.7.5 ships a wheel and is imported as `hnswlib`. Pinned to 0.7.5 because 0.7.6 has no wheel. Its `filter` support in `knn_query` is verified in prompt 1.2.

## D5: extra dependencies beyond section 3
`python-dotenv` (read `.env`), `httpx` (FastAPI TestClient, dev only), `hatchling` (build backend). All are plumbing, not new technologies.
