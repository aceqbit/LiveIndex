# OPEN_QUESTIONS

1. `make` targets are `just` recipes (see DECISIONS D2). `just up` verifies Postgres instead of starting containers.
2. The Qdrant adapter will use the in-process local client (`QdrantClient(path=...)`), which may differ from server behaviour. The README must say so.
3. Prompt 0 asks for a `chunks_baseline` pgvector table; skipped because pgvector is unavailable (D3).
4. Write endpoints (`/v1/docs`, ACL, groups) are not authenticated: the spec defines verified principals for search only. Safest choice taken: search always needs a verified principal; writes are an operator surface and must not be exposed publicly.
5. Tokens are HMAC-signed `<principal>.<sig>` strings using `LIVEINDEX_TOKEN_SECRET`; the spec says "verified token" without a scheme.
