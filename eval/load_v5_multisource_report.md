# V5 Load Test

## Run
- git_commit: `239e76a`
- git_dirty: `True`
- environment_status: `invalid`
- preflight_passed: `True`
- preflight_error_codes: `{}`
- started_at: `2026-08-23T18:20:20.385637+00:00`
- ended_at: `2026-08-23T18:21:21.496745+00:00`
- base url: `http://127.0.0.1:8008`
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
- throughput rps: `1.30`
- successful rps: `1.30`

## Stages

| Workload | C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_pipeline | 1 | 20 | 20 | 0 | 0 | 0.62 | 0.62 | 1620.0 | 1861.9 | 1865.2 | DeepSeek |
| full_pipeline | 4 | 20 | 20 | 0 | 0 | 2.30 | 2.30 | 1580.1 | 1971.0 | 2021.0 | DeepSeek |
| full_pipeline | 8 | 20 | 20 | 0 | 0 | 3.98 | 3.98 | 1679.2 | 2485.6 | 2496.4 | DeepSeek |

## Server Timings

| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_pipeline | 1 | 1356.9 | 320.9 | 247.1 | 3.3 | 1856.2 | 9.3 |
| full_pipeline | 4 | 1435.8 | 548.9 | 280.8 | 3.6 | 1965.3 | 14.4 |
| full_pipeline | 8 | 1484.2 | 818.2 | 283.9 | 4.8 | 2469.3 | 17.4 |

## Error Codes
- full_pipeline C=1: {}
- full_pipeline C=4: {}
- full_pipeline C=8: {}

## Host Metrics
- host CPU/RAM metrics not collected
