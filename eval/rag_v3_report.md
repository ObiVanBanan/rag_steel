# RAG V3 Evaluation

## Run
- commit: `cee953c`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `10`
- collection: `steel_products_active`

## Overall
- status accuracy: `1.0000`
- cannot_process precision/recall: `0.0000` / `0.0000`
- not_found precision/recall: `0.0000` / `0.0000`
- hard violation rate: `0.0000`
- preferred hit@1/@3/@5: `1.0000` / `1.0000` / `1.0000`
- preferred precision@5: `0.9600`
- MRR: `1.0000`
- eligible hit@1/@5: `1.0000` / `1.0000`
- eligible coverage@5: `0.4296`
- LD precision/recall/exact: `1.0000` / `1.0000` / `1.0000`
- invalid competitor rate: `0.0000`
- embedding p50/p95: `271.0` / `2178.0` ms
- qdrant p50/p95: `130.9` / `226.9` ms
- ranking p50/p95: `0.3` / `1.0` ms

## By Category

| Category | Cases | Status Acc | Hard Viol. | Pref hit@5 |
| --- | ---: | ---: | ---: | ---: |
| hard_only | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_plus_material | 3 | 1.0000 | 0.0000 | 1.0000 |

## Failure Stages

- ok: 10
