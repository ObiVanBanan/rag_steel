# V5 Load Test

## Run
- git_commit: `0864d57`
- git_dirty: `True`
- environment_status: `invalid`
- preflight_passed: `True`
- preflight_error_codes: `{}`
- started_at: `2026-08-24T13:22:31.378648+00:00`
- ended_at: `2026-08-24T13:23:33.289540+00:00`
- base url: `http://127.0.0.1:8006`
- endpoint: `/v2/search`
- dataset: `eval\v5_golden_queries.jsonl`
- dataset sha256: `7b3b76da33c323814eefea698294cd0ccdc3148be1f474eb13defdee051a097b`
- qdrant alias: `steel_products_active`
- resolved collection: `steel_products_text-embedding-3-small_20260823T180632Z`
- embedding model: `text-embedding-3-small`
- deepseek model: `deepseek-v4-flash`
- max concurrent searches: `8`
- concurrency levels: `1, 4, 8`
- workloads: `full_pipeline`
- warmup requests: `3`
- requests per stage: `20`
- client timeout seconds: `180.0`

## Preflight
- status: `passed`
- observations: `3`
- error codes: `{}`

## Totals
- total requests: `60`
- success: `60`
- busy (503 SERVICE_BUSY): `0`
- other errors: `0`
- throughput rps: `1.43`
- successful rps: `1.43`

## Stages

| Workload | C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_pipeline | 1 | 20 | 20 | 0 | 0 | 0.70 | 0.70 | 1459.1 | 1643.0 | 1681.0 | DeepSeek |
| full_pipeline | 4 | 20 | 20 | 0 | 0 | 2.35 | 2.35 | 1561.9 | 1906.3 | 1935.0 | DeepSeek |
| full_pipeline | 8 | 20 | 20 | 0 | 0 | 4.31 | 4.31 | 1472.6 | 1939.6 | 2120.3 | DeepSeek |

## Server Timings

| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_pipeline | 1 | 1199.1 | 305.1 | 232.6 | 5.0 | 1637.7 | 13.1 |
| full_pipeline | 4 | 1305.4 | 564.9 | 270.2 | 4.2 | 1885.7 | 23.7 |
| full_pipeline | 8 | 1374.3 | 545.0 | 279.8 | 4.6 | 1906.2 | 54.4 |

## Error Codes
- full_pipeline C=1: {}
- full_pipeline C=4: {}
- full_pipeline C=8: {}

## Host Metrics
- host CPU/RAM metrics not collected
