# E2E V3 Evaluation

## Run
- commit: `dc2d2c5`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `10`
- collection: `steel_products_active`

## Overall
- status accuracy: `1.0000`
- cannot_process precision/recall: `0.0000` / `0.0000`
- not_found precision/recall: `0.0000` / `0.0000`
- e2e preferred hit@1/@5: `1.0000` / `1.0000`
- e2e eligible hit@1/@5: `1.0000` / `1.0000`
- invalid competitor rate: `0.0000`
- overall pass rate: `1.0000`
- strict overall pass rate: `1.0000`
- wall-clock p50/p95/p99: `1598.6` / `2190.3` / `2190.3` ms
- embedding p50/p95: `295.0` / `611.1` ms
- qdrant p50/p95: `120.9` / `161.0` ms
- ranking p50/p95: `0.3` / `2.2` ms

## By Category

| Category | Cases | Status Acc | Overall | Strict |
| --- | ---: | ---: | ---: | ---: |
| hard_only | 7 | 1.0000 | 1.0000 | 1.0000 |
| hard_plus_material | 3 | 1.0000 | 1.0000 | 1.0000 |

## Failure Stages

- ok: 10

## Worst Failures
