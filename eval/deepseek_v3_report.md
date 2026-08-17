# DeepSeek V3 Evaluation

## Run
- commit: `dc2d2c5`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `10`
- model: `deepseek-v4-flash`

## Overall
- brand accuracy: `1.0000`
- brand false positive rate: `0.0000`
- brand false negative rate: `0.0000`
- extraction cases: `10`
- hard exact match rate: `1.0000`
- dn / pn / connection accuracy: `1.0000` / `1.0000` / `1.0000`
- hard hallucination rate: `0.0000`
- soft hallucination rate: `0.0000`
- hard missing rate: `0.0000`
- soft missing rate: `0.0000`
- invalid / timeout / upstream / config / unexpected: `0.0000` / `0.0000` / `0.0000` / `0.0000` / `0.0000`
- latency p50 / p95 / p99: `2401.8` / `2685.3` / `2685.3` ms

## By Category

| Category | Cases | Brand Acc | Hard Exact | Invalid |
| --- | ---: | ---: | ---: | ---: |
| hard_only | 7 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_material | 3 | 1.0000 | 1.0000 | 0.0000 |

## Worst Failures
