# CLAUDE.md: LiveIndex

Save this file as `CLAUDE.md` in the repo root. Read it fully at the start of every session.

## 1. What this project is

LiveIndex has two deliverables.

1. **An engine:** a hybrid search engine (in-house BM25, HNSW vectors, reciprocal rank fusion, ONNX cross-encoder reranker) where every write gets an epoch number and every query reads one consistent snapshot. Deletes, updates and permission changes take effect at the acknowledged epoch, and permission filtering happens inside candidate generation.
2. **A conformance suite (`liveindex-check`):** a stateful property test that drives any retriever through random deletes, updates and permission changes while queries run, and checks six invariants. It runs against LiveIndex, a naive pgvector baseline and Qdrant.

Goal: show that the baseline fails the suite, LiveIndex passes it, and relevance does not suffer (nDCG@10 on BEIR).

Non-goals: a general database, learning-to-rank, multilingual support, a hosted SaaS.

## 2. Rules for you (Claude Code)

1. Never invent a number. Every metric in README, docs or commits comes from a file in `results/` written by code in this repo.
2. Do not add a dependency outside section 3 without recording the reason in `DECISIONS.md` first.
3. List a technology in the README only if it is used in the code.
4. Write the failing test first for each invariant, then the code.
5. If the spec is ambiguous, take the safest choice, record it in `OPEN_QUESTIONS.md`, and continue.
6. Run `make check` before declaring a task done and paste the last 20 lines of its output.
7. Small commits, one concern each. No secrets in the repo; use `.env.example`.
8. The suite must never import engine internals. It talks to systems only through the adapter interface (section 8), so it stays usable on third-party stores.
9. Data: BEIR public sets and synthetic documents only. Never real personal data.

## 3. Stack

| Part | Choice |
|---|---|
| Language, API | Python 3.12, FastAPI, uvicorn, pydantic v2 |
| Lexical index | In-house BM25 (numpy arrays, own tokenizer) |
| Vector index | `hnswlib` (check the installed version supports `filter` in `knn_query` and `mark_deleted`) |
| Embeddings | `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384 dimensions, normalised) |
| Reranker | A small cross-encoder (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) exported to ONNX and run with ONNX Runtime |
| Metadata, log, ACL | PostgreSQL 16 with `psycopg` 3; `pgvector` extension for the baseline |
| Third-party adapter | Qdrant in Docker via `qdrant-client` |
| Tests | pytest, hypothesis (`RuleBasedStateMachine`) |
| Relevance eval | `beir` loader (SciFact, NFCorpus); nDCG@10 and recall@10 implemented in `liveindex/eval/metrics.py` |
| Answer eval (Phase 2) | A small LLM judge through the Groq or Anthropic API, env-configured |
| Observability | OpenTelemetry (Python SDK), Prometheus client, Grafana |
| Packaging | Docker, Docker Compose, GitHub Actions; GCP Cloud Run and Cloud SQL (Phase 2) |

## 4. Repo layout

```
liveindex/
  CLAUDE.md  PROMPTS.md  DECISIONS.md  OPEN_QUESTIONS.md  README.md  Makefile
  docker-compose.yml  .env.example
  src/liveindex/
    api/ (FastAPI app, models)   chunking.py  embed.py
    wal/ (log, epoch)            index/ (bm25.py, hnsw.py, visibility.py, segments.py)
    acl.py  search.py  rerank.py  indexer.py  snapshot.py  telemetry.py
    eval/ (metrics.py, beir_runner.py)
  src/liveindex_check/
    adapter.py  invariants.py  machine.py  report.py  cli.py
    adapters/ (liveindex.py, pgvector_baseline.py, qdrant.py)
  tests/  db/migrations/  results/  infra/  docs/
```

`liveindex_check` must not import from `liveindex`, except inside `adapters/liveindex.py`, which talks to the HTTP API only.

## 5. Concepts and data model

**Epoch.** A global monotonic integer in Postgres. Every write transaction increments it and returns the new value. An acknowledgement carries that epoch, and the change is durable.

**Visibility (MVCC-style).** Each chunk has `born_epoch` and `dead_epoch` (a very large integer while alive). A query at epoch `e` sees chunk `c` iff `born_epoch <= e < dead_epoch` and the chunk's document is allowed for the principal at `e`. An update sets `dead_epoch = e` on the old version's chunks and `born_epoch = e` on the new ones in the same epoch.

```sql
CREATE TABLE epoch_counter (id int PRIMARY KEY CHECK (id = 1), epoch bigint NOT NULL);
CREATE TABLE documents (doc_id text, version int, tenant text, text text,
  born_epoch bigint NOT NULL, dead_epoch bigint NOT NULL DEFAULT 9223372036854775807,
  PRIMARY KEY (doc_id, version));
CREATE TABLE wal (epoch bigint, seq int, op text CHECK (op IN ('upsert','delete','acl_grant','acl_revoke')),
  doc_id text, version int, payload jsonb, PRIMARY KEY (epoch, seq));
CREATE TABLE acl_entries (doc_id text, principal text,   -- principal is a user or a group
  granted_epoch bigint NOT NULL, revoked_epoch bigint NOT NULL DEFAULT 9223372036854775807);
CREATE TABLE group_members (grp text, member text, added_epoch bigint, removed_epoch bigint DEFAULT 9223372036854775807);
```

**In memory (one writer process).** Arrays indexed by an internal integer `chunk_id`: `doc_ix`, `born`, `dead` (numpy int64), the embedding matrix, BM25 postings (`term -> (chunk_ids, tfs)`), doc lengths, and the HNSW index (labels = `chunk_id`). Growth uses preallocated capacity; resize under a lock; the indexer is the only writer; readers capture array lengths at query start.

**Allowed documents at epoch e for a principal.** Expand the principal's groups at `e`, collect documents with an ACL entry valid at `e`, build a boolean mask over documents. Cache per `(principal, acl_epoch)`.

## 6. Write path

1. `POST /v1/docs` (upsert), `DELETE /v1/docs/{id}`, `PUT /v1/docs/{id}/acl`. Chunk (default 300 tokens, 15% overlap, sentence-aware), embed.
2. One Postgres transaction: write the document and WAL rows, `UPDATE epoch_counter SET epoch = epoch + 1 RETURNING epoch`. Return `{epoch}`.
3. The indexer applies WAL rows in epoch order: add chunks (postings and HNSW), set `dead_epoch` on superseded chunks, update ACL state. `index_epoch` is the last fully applied epoch.
4. `GET /v1/epoch` returns `{committed, index_epoch}`. A query may pass `min_epoch`; the API waits (bounded, default 5 s) until `index_epoch >= min_epoch`.
5. A delete also calls `hnswlib.mark_deleted` for each chunk, after setting `dead_epoch`.

## 7. Read path

1. `POST /v1/search` with `{query, k, principal, min_epoch?}`. Principal comes from the verified token or, in tests, a header guarded by `ALLOW_TEST_PRINCIPAL=1`. Never trust a principal in the body.
2. Snapshot: `e = index_epoch` (or `max(index_epoch, min_epoch)` after waiting).
3. Visible mask over chunks: `(born <= e) & (dead > e) & doc_allowed[doc_ix]`.
4. BM25: score only postings in the mask (`k1 = 1.2`, `b = 0.75`, lowercase, regex word tokenizer, small stop-word list), top 100.
5. Vectors: HNSW query with a filter function over the mask (`ef` default 200). If the mask selects fewer than `EXACT_FALLBACK_FRACTION` (default 0.05) of chunks, skip HNSW and run exact dot-product search over the visible chunks.
6. Fuse with reciprocal rank fusion (`k = 60`), take the top 50, rerank with the ONNX cross-encoder (optional flag), return top `k` with `{doc_id, version, chunk_text, score, epoch}`.
7. Response includes the epoch used.

**Compaction.** A background job rebuilds segments without chunks where `dead <= oldest_active_query_epoch`. After compaction, the raw index files must not contain any text of a deleted document (the suite checks this in Phase 2).

## 8. Conformance suite

Adapter interface (`liveindex_check/adapter.py`):

```python
class Adapter(Protocol):
    def reset(self) -> None: ...
    def upsert(self, doc_id: str, text: str, allowed: set[str], tenant: str) -> Ack: ...
    def delete(self, doc_id: str) -> Ack: ...
    def set_acl(self, doc_id: str, allowed: set[str]) -> Ack: ...
    def search(self, principal: str, query: str, k: int, min_ack: Ack | None) -> Result: ...
    def settle(self, ack: Ack, timeout_s: float) -> bool: ...   # waits for visibility if the system has a notion of it
```

`Ack` holds an opaque token and, where the system has epochs, the epoch. `Result` holds a list of `(doc_id, version, text)` and the epoch if known.

**Query trick:** to test visibility, the query is the verbatim text of a chunk of the target document version. Both BM25 and embeddings then retrieve it reliably, so absence checks are meaningful and presence checks are stable. Every generated document contains a unique marker sentence (`zq-<doc>-<version>-<nonce>`).

**Hypothesis state machine** (`machine.py`): a model of the truth (documents, versions, ACLs) plus rules: `upsert_new`, `upsert_update`, `delete`, `grant`, `revoke`, `search_present`, `search_absent`, `concurrent_burst` (queries from threads while a write is in flight, Phase 2). Settings: `max_examples` and `stateful_step_count` from CLI. Failing sequences must shrink; save the minimal sequence to `results/failures/`.

**Invariants** (`invariants.py`):

| # | Invariant |
|---|---|
| I1 | After a delete is acknowledged (and the adapter settled within its stated deadline), no chunk of that document is returned |
| I2 | After an update is acknowledged, no old-version chunk is returned and the new version's text is retrievable |
| I3 | After a revoke is acknowledged, that principal never gets a chunk of that document |
| I4 | Every returned chunk belongs to a document the principal is allowed to see, always, even before settling |
| I5 | Within one response, each document appears in exactly one version |
| I6 | nDCG@10 on the same BEIR set is within 1 point of a static index of the same corpus (a separate relevance test, not part of the state machine) |

Measured outputs, per system: `leak_rate` (I3 and I4 violations per query), `stale_hit_rate` (I1 and I2), `time_to_invisible` (p50, p95, p99, measured from ack to the first clean query), `ops`, `queries`. Report file: `results/suite_<system>_<utc>.json` with `git_sha` and seeds. Exit code 1 if a system marked `--expect pass` violates any invariant; 0 if a system marked `--expect fail` does violate (this keeps the baseline honest).

**Adapters.**
- `liveindex`: HTTP API; `settle` polls `/v1/epoch`.
- `pgvector_baseline`: synchronous inserts; ANN top-k first, then post-filter by ACL; deletes handled by an async worker with `DELETE_LAG_S` (default 2); a `--fixed` flag sets lag 0 and pre-filter to show the suite passes a correct design.
- `qdrant`: payload filter for ACL (`allowed` array), delete API; test with and without `wait=true`.

## 9. Observability

Spans: `ingest.chunk`, `ingest.embed`, `wal.commit`, `index.apply`, `query.snapshot`, `query.acl`, `query.bm25`, `query.ann`, `query.fuse`, `query.rerank`. Attributes: `epoch`, `k`, `visible_fraction`, `fallback_exact` (bool). Prometheus: `liveindex_ops_total{op}`, `liveindex_epoch_lag`, `liveindex_tombstones`, `liveindex_query_seconds` histogram.

## 10. Commands (Makefile)

`make up` · `make down` · `make migrate` · `make test` · `make eval DATASET=scifact` · `make suite SYSTEM=liveindex EXPECT=pass` · `make suite SYSTEM=pgvector EXPECT=fail` · `make suite-all` · `make scale N=100000` · `make bench` · `make lint` · `make check` (lint, tests, a 1,000-operation suite) · `make report` (regenerates README tables from `results/`).

## 11. Definition of done

**Phase 1:** BEIR SciFact and NFCorpus nDCG@10 and recall@10 reported for BM25, vectors and hybrid; the baseline fails I1 and I3 with a shrunk minimal sequence; LiveIndex passes 10,000 operations on I1 to I4; `make check` is green; a manual delete and a manual revoke are invisible at the acknowledged epoch.

**Phase 2:** I5 and I6 implemented; Qdrant adapter works; compaction scan passes; 100,000-chunk run; per-stage latency breakdown from traces; stale-citation rate from the LLM judge; Docker Compose and GitHub Actions gate; GCP deployment plan applies and destroys cleanly.

## 12. Metrics to record (code writes them)

Corpus and chunk counts; operations and queries per run; leak rate; stale-hit rate; time-to-invisible p50, p95, p99 per system; nDCG@10 and recall@10 per retriever and dataset, plus the static-index comparison for I6; retrieval latency p95 with and without the reranker; write throughput under live queries; spans per query; line coverage; judge-based stale-citation rate on 200 questions.

## 13. Resume rules

Claim "built" and "measured". Never "first" or "novel". The unified-data-layer design (one Postgres transaction plus row-level filtering) already exists; describe LiveIndex as an engine plus a reusable suite. Every number matches a file in `results/` and its git SHA.
