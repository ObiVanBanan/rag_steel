# V5 Load Test

## Run
- commit: `0538255`
- started_at: `2026-08-20T15:51:02.999247+00:00`
- ended_at: `2026-08-20T15:52:20.722179+00:00`
- base url: `asgi://main.app`
- endpoint: `/v2/search`
- dataset: `eval\v5_golden_queries.jsonl`
- dataset sha256: `b9b0f7fc497924c2325978546bba87de27a1180993a91ac231b14641012c6af6`
- qdrant alias: `steel_products_active`
- resolved collection: `steel_products_text-embedding-3-small_20260816T095000Z`
- embedding model: `text-embedding-3-small`
- deepseek model: `deepseek-v4-flash`
- max concurrent searches: `8`
- concurrency levels: `1, 2, 5, 10, 20, 50`
- warmup requests: `5`
- requests per stage: `100`
- client timeout seconds: `180.0`

## Totals
- total requests: `600`
- success: `0`
- busy (503 SERVICE_BUSY): `162`
- other errors: `438`
- throughput rps: `176.41`
- successful rps: `0.00`

## Stages

| C | Requests | Success | Busy | Errors | RPS | Successful RPS | p50 | p95 | p99 | Bottleneck |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 100 | 0 | 0 | 100 | 66.02 | 0.00 | 15.5 | 26.9 | 29.3 | unknown |
| 2 | 100 | 0 | 0 | 100 | 207.79 | 0.00 | 6.8 | 27.7 | 31.9 | unknown |
| 5 | 100 | 0 | 0 | 100 | 209.88 | 0.00 | 18.3 | 43.6 | 68.2 | unknown |
| 10 | 100 | 0 | 20 | 80 | 291.02 | 0.00 | 34.4 | 49.7 | 55.0 | unknown |
| 20 | 100 | 0 | 60 | 40 | 344.27 | 0.00 | 50.3 | 91.4 | 95.1 | unknown |
| 50 | 100 | 0 | 82 | 18 | 339.38 | 0.00 | 101.9 | 234.1 | 249.9 | unknown |

## Server Timings

| C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | Server total p95 | External overhead p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Saturation
- C=1: busy=0, 5xx+timeouts=100, degraded=True, readiness_before=200, readiness_after=200
- C=2: busy=0, 5xx+timeouts=100, degraded=True, readiness_before=200, readiness_after=200
- C=5: busy=0, 5xx+timeouts=100, degraded=True, readiness_before=200, readiness_after=200
- C=10: busy=20, 5xx+timeouts=80, degraded=True, readiness_before=200, readiness_after=200
- C=20: busy=60, 5xx+timeouts=40, degraded=True, readiness_before=200, readiness_after=200
- C=50: busy=82, 5xx+timeouts=18, degraded=True, readiness_before=200, readiness_after=200

## Error Codes
- C=1: {"DEEPSEEK_UNAVAILABLE": 100}
- C=2: {"DEEPSEEK_UNAVAILABLE": 100}
- C=5: {"DEEPSEEK_UNAVAILABLE": 100}
- C=10: {"DEEPSEEK_UNAVAILABLE": 80, "SERVICE_BUSY": 20}
- C=20: {"DEEPSEEK_UNAVAILABLE": 40, "SERVICE_BUSY": 60}
- C=50: {"DEEPSEEK_UNAVAILABLE": 18, "SERVICE_BUSY": 82}

## Host Metrics
- host CPU/RAM metrics not collected
