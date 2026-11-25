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

## D6: chunking and embedding run in the indexer, not before the WAL commit
CLAUDE.md section 6 chunks and embeds before the transaction. We store the document text in the WAL and let the indexer chunk and embed on apply (embeddings are cached by text hash). The acknowledgement still means "durable at epoch e"; visibility is gated by `index_epoch`. This keeps the write transaction short and makes replay deterministic.

## D7: group membership changes use the existing WAL ops
The WAL `op` check allows only upsert/delete/acl_grant/acl_revoke. Group add/remove are stored as `acl_grant`/`acl_revoke` rows with `doc_id = ''` and payload `{group, member}`.

## D8: `mark_deleted` and time travel
Per section 6.5 a delete tombstones chunks in the HNSW graph after setting `dead_epoch`. Tombstoned chunks never come back from the graph, so a query at an epoch before the delete finds the old version through BM25 and the exact path only. Time travel is a test feature, not a product guarantee for the vector side. Updates do not tombstone old versions.

## D9: copy-on-grow for readers
Readers take one immutable `View` (epoch + array references + bounds). Growth allocates new arrays (and a new HNSW graph) and swaps the reference, so no reader holds a lock and no resize happens under a running query.

## D10: embedding in tests
Tests use a deterministic hashed bag-of-words embedder (`tests/conftest.py`). Relevance numbers in `results/` use the real bge model.
