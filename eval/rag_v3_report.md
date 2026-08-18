# RAG V3 Evaluation

## Run
- commit: `3baad66`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `90`
- collection: `steel_products_text-embedding-3-small_20260816T095000Z`

## Overall
- status accuracy: `1.0000`
- cannot_process precision/recall: `1.0000` / `1.0000`
- not_found precision/recall: `1.0000` / `1.0000`
- hard violation rate: `0.0000`
- preferred hit@1/@3/@5: `0.9318` / `0.9659` / `0.9886`
- preferred precision@5: `0.8045`
- MRR: `0.9521`
- eligible hit@1/@5: `1.0000` / `1.0000`
- eligible coverage@5: `0.3013`
- LD precision/recall/exact: `1.0000` / `1.0000` / `1.0000`
- invalid competitor rate: `0.0000`
- embedding p50/p95: `286.6` / `322.2` ms
- qdrant p50/p95: `113.2` / `175.6` ms
- ranking p50/p95: `0.4` / `4.9` ms

## By Category

| Category | Cases | Status Acc | Hard Viol. | Pref hit@5 |
| --- | ---: | ---: | ---: | ---: |
| compact_syntax | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_only | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_plus_article | 16 | 1.0000 | 0.0000 | 0.9375 |
| hard_plus_material | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_plus_medium | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_plus_multiple_soft | 7 | 1.0000 | 0.0000 | 1.0000 |
| hard_plus_series | 2 | 1.0000 | 0.0000 | 1.0000 |
| impossible_hard | 1 | 1.0000 | 0.0000 | 0.0000 |
| missing_connection | 7 | 1.0000 | 0.0000 | 1.0000 |
| missing_dn | 7 | 1.0000 | 0.0000 | 1.0000 |
| missing_pn | 7 | 1.0000 | 0.0000 | 1.0000 |
| natural_language | 7 | 1.0000 | 0.0000 | 1.0000 |
| no_brand | 1 | 1.0000 | 0.0000 | 0.0000 |
| russian_alias | 7 | 1.0000 | 0.0000 | 1.0000 |

## Failure Stages

- ok: 90
