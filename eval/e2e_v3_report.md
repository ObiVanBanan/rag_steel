# E2E V3 Evaluation

## Run
- commit: `3baad66`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `90`
- collection: `steel_products_text-embedding-3-small_20260816T095000Z`

## Overall
- status accuracy: `0.9667`
- cannot_process precision/recall: `0.2500` / `1.0000`
- not_found precision/recall: `1.0000` / `1.0000`
- e2e preferred hit@1/@5: `0.8977` / `0.9545`
- e2e eligible hit@1/@5: `0.9659` / `0.9659`
- invalid competitor rate: `0.0000`
- overall pass rate: `0.9667`
- strict overall pass rate: `0.9556`
- wall-clock p50/p95/p99: `1523.6` / `1860.5` / `2165.3` ms
- embedding p50/p95: `287.3` / `315.4` ms
- qdrant p50/p95: `113.7` / `168.7` ms
- ranking p50/p95: `0.5` / `3.0` ms

## By Category

| Category | Cases | Status Acc | Overall | Strict |
| --- | ---: | ---: | ---: | ---: |
| compact_syntax | 7 | 0.8571 | 0.8571 | 0.8571 |
| hard_only | 7 | 1.0000 | 1.0000 | 1.0000 |
| hard_plus_article | 16 | 1.0000 | 1.0000 | 0.9375 |
| hard_plus_material | 7 | 1.0000 | 1.0000 | 1.0000 |
| hard_plus_medium | 7 | 1.0000 | 1.0000 | 1.0000 |
| hard_plus_multiple_soft | 7 | 1.0000 | 1.0000 | 1.0000 |
| hard_plus_series | 2 | 1.0000 | 1.0000 | 1.0000 |
| impossible_hard | 1 | 1.0000 | 1.0000 | 1.0000 |
| missing_connection | 7 | 1.0000 | 1.0000 | 1.0000 |
| missing_dn | 7 | 1.0000 | 1.0000 | 1.0000 |
| missing_pn | 7 | 1.0000 | 1.0000 | 1.0000 |
| natural_language | 7 | 0.8571 | 0.8571 | 0.8571 |
| no_brand | 1 | 1.0000 | 1.0000 | 1.0000 |
| russian_alias | 7 | 0.8571 | 0.8571 | 0.8571 |

## Failure Stages

- brand_gate_failure: 3
- ok: 86
- ranking_failure: 1

## Worst Failures

### brand_gate_failure
- `v3_russian_alias_0068` `also Ду50 Ру40` expected `exact_match` actual `cannot_process`
- `v3_compact_syntax_0075` `ALSO DN50PN40` expected `exact_match` actual `cannot_process`
- `v3_natural_language_0082` `Подбери мне аналог ALSO на диаметр 50 давление 40 бар штуцерное сталь 09г2с жидкость` expected `exact_match` actual `cannot_process`

### ranking_failure
- `v3_hard_plus_article_0027` `MARSHAL DN40 PN40 фланцевое Цф.00.1.040.040` expected `exact_match` actual `exact_match`
