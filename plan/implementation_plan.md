# Production Wave: Observability + Index Safety

## General Information
- Repository: `ObiVanBanan/rag_steel`
- Branch: `codex/v2-eval-pipeline`
- Goal: add request-level observability, Prometheus-style metrics, and Qdrant index compatibility checks without changing search quality or ranking semantics.
- Constraints: preserve existing search pipeline order, avoid unrelated local changes, keep legacy indexes readable during migration.

## Roadmap by Milestones

### Milestone 1: Observability plumbing
- [ ] Task 1.1 Add request ID middleware and response header support
  - Files: `main.py`, `src/rag_steel/observability.py`, `tests/test_main_api.py`
  - DoD: every HTTP request gets a stable request ID, `X-Request-ID` is echoed to the client, and the ID is available to downstream code via request context.
- [ ] Task 1.2 Add structured request and search completion logging
  - Files: `main.py`, `src/rag_steel/search_engine.py`, `src/rag_steel/observability.py`, `tests/test_main_api.py`
  - DoD: request completion and search completion emit machine-readable JSON logs with request ID, method/path/status, and safe timing/result fields only.
- [ ] Task 1.3 Expose Prometheus-compatible metrics endpoint
  - Files: `main.py`, `src/rag_steel/observability.py`, `tests/test_main_api.py`
  - DoD: `GET /metrics` returns exposition text with HTTP and pipeline metrics; labels stay low-cardinality and exclude request/query/article values.

### Milestone 2: Index safety and readiness
- [ ] Task 2.1 Add index metadata schema helpers and compatibility checks
  - Files: `src/rag_steel/index_metadata.py`, `src/rag_steel/search_engine.py`, `tests/test_search_engine.py`
  - DoD: startup/readiness can validate collection metadata, vector dimension, embedding model, and supported index format version with stable machine-readable failure reasons.
- [ ] Task 2.2 Persist richer build metadata during indexing
  - Files: `src/rag_steel/indexer.py`, `tests/test_indexer.py`
  - DoD: new collections store build metadata including schema/version, embedding info, dataset hash, build timestamp, git commit, source row count, and point count.
- [ ] Task 2.3 Expand readiness payloads for operators
  - Files: `main.py`, `src/rag_steel/search_engine.py`, `tests/test_main_api.py`
  - DoD: readiness returns explicit qdrant/index sections and remains permissive for legacy indexes when metadata is absent but vector dimensions still match.

### Milestone 3: Verification and release hygiene
- [ ] Task 3.1 Add focused unit coverage for metrics, request ID, and compatibility cases
  - Files: `tests/test_main_api.py`, `tests/test_search_engine.py`, `tests/test_indexer.py`
  - DoD: tests cover generated/existing request IDs, `/metrics`, cardinality guards, matching/mismatching metadata, and legacy migration behavior.
- [ ] Task 3.2 Run full quality gates and regression smoke
  - Files: `eval/*`, `tests/*`
  - DoD: `uv run pytest -q`, Ruff, and `uv run python -m eval.evaluate_rag_v3 --max-cases 10` pass without search-quality regressions.

## Dependencies Between Tasks
- Task 1.1 should land before Task 1.2 and Task 1.3 so logs and metrics can read request context.
- Task 2.1 should land before Task 2.3 so readiness can consume the new compatibility helper.
- Task 2.2 should land before Task 2.3 so the readiness payload reflects the new metadata shape.
- Task 3.1 depends on both milestones 1 and 2.

## Risks and Notes
- Keep metrics labels fixed and low-cardinality.
- Do not add tracing, queues, dashboards, or prompt/model changes in this wave.
- Do not touch `docs/plan.md` or `fix.md`.
- Preserve existing search and ranking results; any differences should be treated as regressions.
