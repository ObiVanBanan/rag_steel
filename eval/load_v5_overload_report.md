# V5 Load Test

## Run
- git_commit: `cbdbacc`
- git_dirty: `True`
- environment_status: `invalid`
- preflight_passed: `True`
- preflight_error_codes: `{}`
- started_at: `2026-08-20T16:37:08.716000+00:00`
- ended_at: `2026-08-20T16:37:28.437424+00:00`
- base url: `http://127.0.0.1:8006`
- endpoint: `/v2/search`
- dataset: `eval\v5_golden_queries.jsonl`
- dataset sha256: `b9b0f7fc497924c2325978546bba87de27a1180993a91ac231b14641012c6af6`
- qdrant alias: `steel_products_active`
- resolved collection: `steel_products_text-embedding-3-small_20260816T095000Z`
- embedding model: `text-embedding-3-small`
- deepseek model: `deepseek-v4-flash`
- max concurrent searches: `8`
- concurrency levels: `10, 20, 50`
- workloads: `full_pipeline`
- warmup requests: `5`
- requests per stage: `50`
- client timeout seconds: `180.0`

## Preflight
- status: `passed`
- observations: `3`
- error codes: `{}`

## Totals
- total requests: `150`
- success: `23`
- busy (503 SERVICE_BUSY): `126`
- other errors: `1`
- throughput rps: `24.36`
- successful rps: `3.74`

## Stages

| Workload | C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_pipeline | 10 | 50 | 7 | 42 | 1 | 23.07 | 3.23 | 13.2 | 2063.5 | 2151.6 | DeepSeek |
| full_pipeline | 20 | 50 | 8 | 42 | 0 | 25.23 | 4.04 | 135.6 | 1856.0 | 1931.0 | DeepSeek |
| full_pipeline | 50 | 50 | 8 | 42 | 0 | 24.90 | 3.98 | 475.2 | 1843.2 | 1987.7 | DeepSeek |

## Server Timings

| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_pipeline | 10 | 1600.0 | 569.2 | 301.4 | 5.0 | 2091.6 | 68.6 |
| full_pipeline | 20 | 1376.3 | 533.5 | 235.9 | 4.9 | 1863.4 | 90.7 |
| full_pipeline | 50 | 1423.7 | 290.1 | 301.7 | 7.7 | 1861.0 | 156.7 |

## Error Codes
- full_pipeline C=10: {"EMBEDDING_UNAVAILABLE": 1, "SERVICE_BUSY": 42}
- full_pipeline C=20: {"SERVICE_BUSY": 42}
- full_pipeline C=50: {"SERVICE_BUSY": 42}

## Host Metrics
- host CPU/RAM metrics not collected
