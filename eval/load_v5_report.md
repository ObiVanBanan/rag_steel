# V5 Load Test

## Run
- git_commit: `526d8bf`
- git_dirty: `True`
- environment_status: `invalid`
- preflight_passed: `True`
- preflight_error_codes: `{}`
- started_at: `2026-08-20T16:17:17.932762+00:00`
- ended_at: `2026-08-20T16:23:51.283888+00:00`
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
- success: `495`
- busy (503 SERVICE_BUSY): `0`
- other errors: `5`
- throughput rps: `1.37`
- successful rps: `1.36`

## Stages

| Workload | C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_pipeline | 1 | 100 | 99 | 0 | 1 | 0.59 | 0.58 | 1648.9 | 1952.7 | 3324.7 | DeepSeek |
| full_pipeline | 2 | 100 | 100 | 0 | 0 | 1.18 | 1.18 | 1656.8 | 2047.4 | 2206.0 | DeepSeek |
| full_pipeline | 4 | 100 | 100 | 0 | 0 | 2.25 | 2.25 | 1702.5 | 2273.9 | 2391.2 | DeepSeek |
| full_pipeline | 6 | 100 | 99 | 0 | 1 | 2.88 | 2.85 | 1854.5 | 3252.8 | 3503.9 | DeepSeek |
| full_pipeline | 8 | 100 | 97 | 0 | 3 | 3.29 | 3.20 | 1865.8 | 4151.5 | 4755.0 | DeepSeek |

## Server Timings

| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_pipeline | 1 | 1416.5 | 585.2 | 178.2 | 4.9 | 1953.9 | 12.6 |
| full_pipeline | 2 | 1474.3 | 583.2 | 174.1 | 4.8 | 1909.4 | 12.3 |
| full_pipeline | 4 | 1500.4 | 587.3 | 184.5 | 4.9 | 2005.3 | 25.0 |
| full_pipeline | 6 | 1700.3 | 585.4 | 304.5 | 5.6 | 2339.1 | 375.6 |
| full_pipeline | 8 | 1937.0 | 606.9 | 398.0 | 5.9 | 2302.6 | 624.9 |

## Error Codes
- full_pipeline C=1: {"EMBEDDING_TIMEOUT": 1}
- full_pipeline C=2: {}
- full_pipeline C=4: {}
- full_pipeline C=6: {"EMBEDDING_UNAVAILABLE": 1}
- full_pipeline C=8: {"EMBEDDING_TIMEOUT": 2, "EMBEDDING_UNAVAILABLE": 1}

## Host Metrics
- host CPU/RAM metrics not collected
