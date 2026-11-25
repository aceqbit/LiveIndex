# PROMPTS.md: LiveIndex build prompts for Claude Code

Paste the prompts in order. One prompt per session: run `/clear` first, work on a branch named after the prompt (`p1-3-epoch`), commit, merge when the acceptance checks pass.

Before prompt 0: create an empty repo, copy `CLAUDE.md` (from `liveindex_CLAUDE.md`) into its root, and install Docker, Python 3.12 and `uv`.

Each prompt has: **Read**, **Do**, **Constraints**, **Accept** (commands that must pass), **Report**. If an accept check fails, fix it before reporting. Report in the format given, with no extra prose.

---

## Prompt 0: bootstrap

**Read:** `CLAUDE.md` in full.

**Do:**
1. Create the layout from section 4 with empty modules, a `uv` project with `pytest`, `ruff`, `mypy`, the Makefile targets from section 10 (stubs exit 1 except `lint` and `test`), `docker-compose.yml` with `postgres` (pgvector image), `qdrant`, `.env.example`, `.gitignore`, `DECISIONS.md`, `OPEN_QUESTIONS.md`.
2. Put the SQL from section 5 in `db/migrations/001_init.sql` plus a migration runner for `make migrate`. Add the pgvector extension and a `chunks_baseline` table for the baseline (`chunk_id bigserial, doc_id text, version int, text text, allowed text[], embedding vector(384)`).
3. GitHub Actions `ci.yml` running `make lint` and `make test`.
4. README with a one-paragraph description and these headings only: Problem, Architecture, Quick start, Results (generated), Limits.

**Constraints:** no application logic yet; no dependency outside section 3.

**Accept:** `make up && make migrate && make lint && make test` pass.

**Report:** file tree (depth 2) and command output.

---

## Phase 1

### Prompt 1.1: data, metrics, baseline numbers

**Read:** `CLAUDE.md` sections 3, 8, 12.

**Do:**
1. `eval/beir_runner.py`: load BEIR SciFact and NFCorpus with the `beir` loader into a local cache folder (not committed).
2. `eval/metrics.py`: implement nDCG@10 and recall@10 from scratch with unit tests on hand-computed examples.
3. `chunking.py`: sentence-aware chunker, 300 tokens target, 15% overlap, tokens counted with the embedding model's tokenizer. Tests: no empty chunks, overlap respected, deterministic.
4. `embed.py`: batch embedding with `bge-small-en-v1.5`, normalised vectors, a disk cache keyed by text hash.
5. A simple pgvector-only retriever (`eval/pgvector_only.py`) and run it on both sets; write `results/beir_pgvector_<set>_<utc>.json`.

**Constraints:** documents in SciFact and NFCorpus are short; chunk only when above the target size. Record chunk counts.

**Accept:** `make eval DATASET=scifact` writes a results file with nDCG@10 and recall@10.

**Report:** the numbers and the chunk counts, and one line on whether they look sane for a small bi-encoder.

### Prompt 1.2: in-house BM25 and the engine core

**Read:** `CLAUDE.md` sections 5 and 7.

**Do:**
1. `index/bm25.py`: tokenizer, postings as numpy arrays, `add_chunks`, `score(query, mask, top_n)`, `k1 = 1.2`, `b = 0.75`. Property tests against a slow reference implementation on random corpora.
2. `index/hnsw.py`: a wrapper around `hnswlib` with `add`, `mark_deleted`, `knn(query, k, filter_fn)`, and the exact-search fallback over a mask. Verify in a test that the installed `hnswlib` supports `filter` and `mark_deleted`; if not, record it in `DECISIONS.md` and implement the filter by oversampling and masking with a recall check.
3. `search.py`: reciprocal rank fusion (`k = 60`).
4. `api/`: FastAPI `POST /v1/docs`, `POST /v1/search`, in-memory only for now (no epochs yet).
5. Evaluate BM25, vectors and hybrid on SciFact and NFCorpus; write results files.

**Constraints:** BM25 correctness first. A test must compare your top-10 ranking with the reference on at least 20 random queries.

**Accept:** `make test` passes; `make eval DATASET=scifact RETRIEVER=hybrid` writes results and hybrid nDCG@10 is not below the best single retriever (if it is, investigate and record why in `docs/notes.md`).

**Report:** a table of nDCG@10 and recall@10 for bm25, vectors, hybrid on both sets.

### Prompt 1.3: epochs, log and visibility

**Read:** `CLAUDE.md` sections 5 and 6.

**Do:**
1. `wal/`: the single transaction that writes the document, WAL rows, and increments `epoch_counter`; return the epoch.
2. `indexer.py`: apply WAL rows in epoch order; maintain `born`, `dead`, `doc_ix` arrays; `index_epoch`; recovery by replaying the log from epoch 0 on start (snapshots come in Phase 2).
3. `index/visibility.py`: build the visible-chunk mask for an epoch.
4. `GET /v1/epoch`, `min_epoch` waiting in search, and the epoch in every response.
5. Tests: update sets `dead` on old chunks and `born` on new in the same epoch; a search at an older epoch still sees the old version (time travel works); restart replays to the same state.

**Constraints:** one writer; readers capture array lengths; resize under a lock; no reader holds a lock during search.

**Accept:** a test ingests a document, deletes it, and a query at the delete epoch returns nothing while a query at the previous epoch returns it.

**Report:** the epoch sequence of that test and the restart replay time on SciFact.

### Prompt 1.4: ACL inside the search

**Read:** `CLAUDE.md` sections 5 and 7.

**Do:**
1. `acl.py` and `PUT /v1/docs/{id}/acl`: entries with `granted_epoch` and `revoked_epoch`; group membership with epochs; `allowed_docs(principal, epoch)` returning a boolean mask over documents; cache by `(principal, acl_epoch)`.
2. Apply the mask in BM25 scoring and in HNSW filtering; add the exact-search fallback when the mask selects under 5% of chunks.
3. Principal from a verified token; for tests, a header enabled only when `ALLOW_TEST_PRINCIPAL=1`.
4. Tests: revoke at epoch `e` hides the document at `e` and later but not at `e - 1`; group removal does the same; a user in two groups sees the union; with a tiny allowed set the exact fallback returns the right top-k (compare with brute force).

**Constraints:** never filter after top-k anywhere in the code. Add a test that fails if `search.py` imports a post-filter helper (grep-based test is acceptable).

**Accept:** `make test` passes including the fallback test.

**Report:** the visible fraction and recall of filtered HNSW versus exact search at filter selectivities 50%, 10%, 2%.

### Prompt 1.5: conformance suite core and adapter interface

**Read:** `CLAUDE.md` section 8.

**Do:**
1. `liveindex_check/adapter.py`, `invariants.py` (I1 to I4), `machine.py` (Hypothesis `RuleBasedStateMachine` with `upsert_new`, `upsert_update`, `delete`, `grant`, `revoke`, `search_present`, `search_absent`), `report.py`, `cli.py`.
2. Document generator: random sentences plus a unique marker sentence per `(doc, version)`; the query is the verbatim text of a chunk.
3. Failing sequences shrink and are written to `results/failures/<system>_<utc>.json`.
4. `adapters/liveindex.py` over the HTTP API.
5. Unit-test the suite with a tiny in-memory fake adapter that is correct (passes) and one that is deliberately wrong in each of the four ways (fails exactly the matching invariant).

**Constraints:** `liveindex_check` must not import `liveindex`. A test enforces this.

**Accept:** `make suite SYSTEM=liveindex EXPECT=pass` runs 1,000 operations and passes; the four broken fake adapters each fail the right invariant.

**Report:** the output of the fake-adapter tests, which shows the suite can fail.

### Prompt 1.6: the baseline the suite must fail

**Read:** `CLAUDE.md` section 8, adapters.

**Do:** implement `adapters/pgvector_baseline.py` backed by the `chunks_baseline` table: ANN top-k with `<=>` first, then ACL post-filter in Python; a worker deleting rows after `DELETE_LAG_S`. Add a `--fixed` mode (lag 0, ACL in the `WHERE` clause) that should pass. Run the suite with `--expect fail` on the default baseline and `--expect pass` on `--fixed`.

**Constraints:** do not make the baseline artificially broken beyond the two real-world defaults (post-filter, async delete). Document both in the adapter docstring.

**Accept:** `make suite SYSTEM=pgvector EXPECT=fail` exits 0 and the report lists I1 and I3 violations with a shrunk sequence; `--fixed` passes.

**Report:** the shrunk minimal failing sequence in plain steps.

### Prompt 1.7: run LiveIndex to 10,000 operations

**Do:** run `make suite SYSTEM=liveindex EXPECT=pass` with 10,000 operations across several seeds. Fix every violation found, each with a regression test that fails first. Record `leak_rate`, `stale_hit_rate` and `time_to_invisible` for both systems.

**Accept:** zero violations on I1 to I4 for LiveIndex over 10,000 operations; the baseline numbers are recorded.

**Report:** a table of both systems' leak rate, stale-hit rate and time-to-invisible (p50, p95, p99), each from a results file.

### Prompt 1.8: CI gate, Compose, README

**Do:** make `docker-compose.yml` start the API, indexer, Postgres, Qdrant, OpenTelemetry Collector, Prometheus and Grafana with health checks. Extend `ci.yml`: lint, tests, and a 1,000-operation suite on every pull request. README: problem table (stale delete, stale update, stale permission), architecture diagram in Mermaid, quick start, how to read a suite report, limits. Add a branch `demo/post-filter` that moves ACL filtering after top-k; confirm CI fails; record the run URL in `docs/ci-proof.md`.

**Accept:** a fresh clone passes `make up && make check`; the demo branch fails CI.

**Report:** the CI run URLs.

---

## Phase 2

### Prompt 2.1: invariants I5, I6 and compaction

**Do:**
1. I5: a rule that runs a search while an update is in flight and checks that each document appears in one version only.
2. I6: a separate relevance test comparing nDCG@10 on SciFact and NFCorpus of the engine after a random mutation schedule against a static index built from the final corpus; threshold 1 point.
3. Compaction: rebuild segments without dead chunks; after it, scan raw index files for the text of every deleted document and fail if found.
4. `concurrent_burst` rule: several query threads during a write.

**Accept:** `make suite SYSTEM=liveindex EXPECT=pass` includes I5, I6 and the compaction scan and passes.

**Report:** the I6 numbers and the compaction scan result.

### Prompt 2.2: Qdrant adapter and third-system report

**Do:** implement `adapters/qdrant.py` using payload filters for ACL and the delete API, with and without `wait=true`. Run all three systems and write `results/suite_all_<utc>.json` with a comparison table (leak rate, stale-hit rate, time-to-invisible).

**Constraints:** report what the suite finds even if Qdrant passes; do not tune it to fail.

**Accept:** `make suite-all` produces the comparison.

**Report:** the comparison table.

### Prompt 2.3: scale to 100,000 chunks and latency

**Do:** a synthetic corpus generator reaching 100,000 chunks; `make scale N=100000` ingests and runs the suite subset and a query benchmark (retrieval p50, p95 without the reranker; write throughput under live queries). Profile with the spans and fix the top hotspot (for example mask building or posting merges). Save before and after.

**Accept:** results files for before and after with the git SHA.

**Report:** a table of before and after, and what changed.

### Prompt 2.4: ONNX reranker

**Do:** export the cross-encoder to ONNX, run it with ONNX Runtime on CPU, rerank the top 50, and measure nDCG@10 and latency with and without it on both BEIR sets. Add an INT8 dynamic-quantized variant and measure the change.

**Constraints:** keep the reranker behind a request flag; default off for the suite.

**Accept:** results file with nDCG@10 and p95 latency for: no rerank, FP32 rerank, INT8 rerank.

**Report:** the table.

### Prompt 2.5: observability

**Do:** OpenTelemetry spans from section 9 with attributes, Prometheus metrics, and a Grafana dashboard JSON (committed) showing epoch lag, tombstones, query latency by stage and time-to-invisible. Add a test that a query produces all expected spans.

**Accept:** a dashboard renders locally with data from `make suite`.

**Report:** the list of panels and the per-stage p95 from a benchmark run.

### Prompt 2.6: stale-citation rate with an LLM judge

**Do:** build 200 questions from the BEIR corpus; for each, mutate the corpus so a document used to answer it is updated or deleted; run retrieval plus a small LLM answer step against the baseline and against LiveIndex; a judge call checks whether the answer cites or uses stale or removed content. Cache all model calls on disk keyed by input. Save `results/judge_<utc>.json` with the stale-citation rate per system and the number of questions.

**Constraints:** model names and keys come from the environment; the harness works offline from the cache for tests. Record judge disagreements by sampling 20 and manually labelling them in `docs/judge_check.md` (leave the labels blank for the user).

**Accept:** results file exists; tests run offline.

**Report:** the stale-citation rate for both systems and the cost in tokens.

### Prompt 2.7: snapshots, packaging, deployment

**Do:**
1. Index snapshots: write postings, HNSW file, arrays and `index_epoch` to a snapshot directory every N minutes; on start, load the latest snapshot and replay the log from its epoch. Measure restart time with and without a snapshot.
2. Package `liveindex_check` as a pip package with its own README and an example for a third-party store.
3. Terraform in `infra/gcp`: Cloud Run (min instances 1, max 1), Cloud SQL Postgres, Secret Manager, Artifact Registry, traces to Cloud Trace; document an AWS variant in `docs/aws.md`. Do not apply; produce `terraform plan` and stop.
4. Image with model weights baked in.

**Accept:** `terraform validate` and `plan` succeed; restart-time results exist.

**Report:** restart times and the plan's resource list.

### Prompt 2.8: write-up and claim check

**Do:** `docs/WRITEUP.md`: problem, design, invariants, results tables generated from `results/`, limits, prior art (the Postgres unified data layer paper, RAG index drift write-ups, vector databases) with the differentiator in one sentence. Add `scripts/verify_claims.py` that extracts every number from README and WRITEUP and confirms each exists in a results file. Regenerate README Results.

**Constraints:** do not use "first" or "novel".

**Accept:** `verify_claims.py` passes; `make check` passes.

**Report:** final results summary and open questions.

---

## Utility prompts

### U1: review for correctness (after each phase)

Review `src/liveindex` against `CLAUDE.md` sections 5 to 7. List every place where a reader could observe a half-applied write, where a filter runs after top-k, where a lock is held during search, and where the epoch used in a response could differ from the one used for filtering. For each finding give file and line, a failing test, and a fix. Do not change code until I approve the list.

### U2: debug a suite failure

The suite reported this failure: `{paste the shrunk sequence}`. Reproduce it deterministically with the seed, print the epoch, visible mask counts and ACL state at each step, explain the cause in five lines, add a regression test that fails first, fix it, and show the suite passing.

### U3: resume bullets from results only

Read `results/` and `README.md`. Write three resume bullets for LiveIndex, each under 30 words, using only numbers found in `results/` (put the file name in brackets after each for me to check, then I will remove the brackets). Use these exact terms where true: hybrid search, reranking, vector database, RAG, ONNX Runtime, OpenTelemetry, Docker, CI/CD, GCP. Do not use "first", "novel" or "cutting-edge".

### U4: interview preparation

Write 15 interview questions a senior engineer would ask about this project (epochs, why in-memory, filtered HNSW recall, exact fallback, what the suite cannot detect, scale-out) with a 4-line model answer each, grounded in this repo's code and results.
