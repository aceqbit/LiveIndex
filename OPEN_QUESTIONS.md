# OPEN_QUESTIONS

1. `make` targets are `just` recipes (see DECISIONS D2). `just up` verifies Postgres instead of starting containers.
2. The Qdrant adapter will use the in-process local client (`QdrantClient(path=...)`), which may differ from server behaviour. The README must say so.
3. Prompt 0 asks for a `chunks_baseline` pgvector table; skipped because pgvector is unavailable (D3).
4. Write endpoints (`/v1/docs`, ACL, groups) are not authenticated: the spec defines verified principals for search only. Safest choice taken: search always needs a verified principal; writes are an operator surface and must not be exposed publicly.
5. Tokens are HMAC-signed `<principal>.<sig>` strings using `LIVEINDEX_TOKEN_SECRET`; the spec says "verified token" without a scheme.
6. I4 ("always, even before settling") is checked two ways: a returned document must be one the principal has been granted at some point (every query, settled or not), and, when the response carries an epoch at or past the last ACL change, the current ACL is enforced strictly. A query that races an unsettled revoke on a system without epochs is not flagged as I4; once settled it is I3.
7. `ops` in a report counts acknowledged writes; `queries` counts searches. "1,000 operations" is met when ops + queries >= 1,000 (Hypothesis ends examples early, so `--ops` is a budget, not an exact count).
8. The suite measures first (violations counted, nothing raised), then, only if something was violated, re-runs with raising on violation so Hypothesis shrinks a minimal sequence. Rates come from the measuring pass only.
