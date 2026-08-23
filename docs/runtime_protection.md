# Runtime Protection Tuning

## Scope

This wave tuned runtime protection only. It does not change search semantics, RAG logic, prompt behavior, brand resolution, hard constraints, or retrieval strategy.

## Effective Runtime Defaults

- `MAX_CONCURRENT_SEARCHES=8` unchanged
- `DEEPSEEK_TIMEOUT_SECONDS=10` was `60`
- `OPENAI_TIMEOUT_SECONDS=10` was `60`
- `QDRANT_TIMEOUT_SECONDS=5` unchanged
- `UPSTREAM_MAX_ATTEMPTS=2` unchanged
- `UPSTREAM_RETRY_BASE_DELAY_SECONDS=0.25` unchanged

## Retry Semantics

- DeepSeek timeout now fails fast with `DEEPSEEK_TIMEOUT`
- OpenAI embeddings timeout now fails fast with `EMBEDDING_TIMEOUT`
- Qdrant timeout now fails fast with `SEARCH_BACKEND_TIMEOUT`
- Retry remains enabled for transient upstream status codes and request errors
- Non-retryable upstream statuses such as `401` still fail immediately

## Guardrails Verified

- search gate remains `8`
- timeout paths no longer spend a second full attempt budget after an upstream timeout
- semaphore slot is released after both success and runtime failures
- `SERVICE_BUSY` behavior and `Retry-After: 1` behavior remain unchanged

## Validation

- targeted fault tests:
  - `uv run pytest -q tests/test_attribute_extractor.py tests/test_config.py tests/test_search_engine.py tests/test_main_api.py`
  - result: `114 passed`
- full test suite:
  - `uv run pytest -q`
  - result: `336 passed`
- lint:
  - `uv run ruff check src eval tests main.py`
  - result: passed

## New Fault Coverage

- DeepSeek timeout does not retry
- DeepSeek `401` does not retry
- OpenAI embeddings timeout does not retry
- OpenAI embeddings `401` does not retry
- Qdrant timeout does not retry
- gate slot is released after failed and successful requests

## Notes

- No request-level overall deadline was added in this wave because the current search path is synchronous and already protected by stage-level timeouts.
- No queueing or gate-size change was introduced.
