# V5 Load Test

## Run
- git_commit: `cbdbacc`
- git_dirty: `False`
- environment_status: `valid`
- preflight_passed: `True`
- preflight_error_codes: `{}`
- started_at: `2026-08-20T16:30:44.742208+00:00`
- ended_at: `2026-08-20T16:36:50.897691+00:00`
- base url: `http://127.0.0.1:8006`
- endpoint: `/v2/search`
- dataset: `eval\v5_golden_queries.jsonl`
- dataset sha256: `b9b0f7fc497924c2325978546bba87de27a1180993a91ac231b14641012c6af6`
- qdrant alias: `steel_products_active`
- resolved collection: `steel_products_text-embedding-3-small_20260816T095000Z`
- embedding model: `text-embedding-3-small`
- deepseek model: `deepseek-v4-flash`
- max concurrent searches: `8`
- concurrency levels: `1, 2, 4, 6, 8`
- workloads: `full_pipeline`
- warmup requests: `5`
- requests per stage: `100`
- client timeout seconds: `180.0`

## Preflight
- status: `passed`
- observations: `3`
- error codes: `{}`

## Totals
- total requests: `500`
- success: `500`
- busy (503 SERVICE_BUSY): `0`
- other errors: `0`
- throughput rps: `1.49`
- successful rps: `1.49`

## Stages

| Workload | C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_pipeline | 1 | 100 | 100 | 0 | 0 | 0.61 | 0.61 | 1587.7 | 1909.4 | 2268.7 | DeepSeek |
| full_pipeline | 2 | 100 | 100 | 0 | 0 | 1.23 | 1.23 | 1616.6 | 1901.8 | 2015.4 | DeepSeek |
| full_pipeline | 4 | 100 | 100 | 0 | 0 | 2.44 | 2.44 | 1589.9 | 1983.6 | 2140.1 | DeepSeek |
| full_pipeline | 6 | 100 | 100 | 0 | 0 | 3.57 | 3.57 | 1620.9 | 1946.4 | 2168.8 | DeepSeek |
| full_pipeline | 8 | 100 | 100 | 0 | 0 | 4.57 | 4.57 | 1635.8 | 2232.8 | 2516.0 | DeepSeek |

## Server Timings

| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_pipeline | 1 | 1417.0 | 390.7 | 175.1 | 4.5 | 1898.4 | 11.3 |
| full_pipeline | 2 | 1388.3 | 571.0 | 179.6 | 4.8 | 1839.1 | 20.5 |
| full_pipeline | 4 | 1378.0 | 568.4 | 233.8 | 5.4 | 1982.8 | 24.4 |
| full_pipeline | 6 | 1416.6 | 443.7 | 264.9 | 6.1 | 1891.9 | 86.8 |
| full_pipeline | 8 | 1433.8 | 556.7 | 286.9 | 5.3 | 1980.8 | 220.8 |

## Error Codes
- full_pipeline C=1: {}
- full_pipeline C=2: {}
- full_pipeline C=4: {}
- full_pipeline C=6: {}
- full_pipeline C=8: {}

## Host Metrics
- host CPU/RAM metrics not collected
