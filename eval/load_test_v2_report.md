# V2 Load Test

## Run
- base url: `http://127.0.0.1:8001`
- dataset: `eval\v2_queries.jsonl`
- limit: `5`
- total requests: `130`

## Totals
- success: `30`
- busy (503): `34`
- timeout: `0`
- other errors: `66`
- average p95 client latency across levels: `118.7` ms

## Levels

| Concurrency | Requests | RPS | Client p50 | Client p95 | Client p99 | Server p50 | Server p95 | Busy | Timeout | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 84.00 | 11.1 | 14.1 | 22.3 | 0.0 | 0.0 | 0 | 0 | 13 |
- concurrency `1` examples: DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction
| 5 | 20 | 119.54 | 36.8 | 47.0 | 48.9 | 0.0 | 0.0 | 0 | 0 | 13 |
- concurrency `5` examples: DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction
| 10 | 20 | 113.18 | 77.3 | 87.1 | 88.6 | 0.0 | 0.0 | 2 | 0 | 12 |
- concurrency `10` examples: SERVICE_BUSY: Search service is temporarily busy; SERVICE_BUSY: Search service is temporarily busy; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction
| 20 | 20 | 131.45 | 103.0 | 123.7 | 136.4 | 0.0 | 0.0 | 4 | 0 | 12 |
- concurrency `20` examples: DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction; DEEPSEEK_CONFIGURATION_MISSING: DeepSeek is required for V2 attribute extraction
| 50 | 50 | 139.71 | 241.0 | 321.5 | 338.0 | 0.0 | 0.0 | 28 | 0 | 16 |
- concurrency `50` examples: SERVICE_BUSY: Search service is temporarily busy; SERVICE_BUSY: Search service is temporarily busy; SERVICE_BUSY: Search service is temporarily busy

## Status Mix
- `1`: {"HTTP_503": 13, "cannot_process": 7}
- `5`: {"HTTP_503": 13, "cannot_process": 7}
- `10`: {"HTTP_503": 14, "cannot_process": 6}
- `20`: {"HTTP_503": 16, "cannot_process": 4}
- `50`: {"HTTP_503": 44, "cannot_process": 6}
