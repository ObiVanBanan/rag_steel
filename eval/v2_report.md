# V2 Search Evaluation

## Run
- commit: `0c937e9`
- collection: `steel_products_active`
- point count: `16016`
- dataset: `eval\v2_queries.jsonl`
- query count: `20`

## Overall
- constraint exact match: `0.8750`
- retrieval any-hit: `0.8125`
- source candidate recall: `0.7539`
- competitor hit@1: `0.6875`
- competitor hit@5: `0.8125`
- competitor precision@5: `0.6615`
- competitor coverage@5: `0.5875`
- invalid competitor rate: `0.3857`
- false exact match rate: `0.2500`
- not-found precision/recall: `1.0000` / `0.7500`
- LD precision/recall: `0.9038` / `1.0000`
- p50/p95 latency total: `421.2` / `980.2` ms

## By Category

| Category | Cases | Parser Exact | Status Accuracy | Hit@5 | Invalid Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| article_exact | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| article_partial | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| brand_dn_pn | 2 | 1.0000 | 1.0000 | 1.0000 | 0.3000 |
| brand_dn_pn_compact | 2 | 1.0000 | 1.0000 | 1.0000 | 0.4000 |
| brand_dn_pn_connection | 2 | 1.0000 | 1.0000 | 1.0000 | 0.4000 |
| brand_dn_pn_material | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| brand_du_ru | 2 | 1.0000 | 1.0000 | 1.0000 | 0.4000 |
| name_exact | 1 | 0.0000 | 1.0000 | 1.0000 | 0.8000 |
| natural_language | 2 | 0.0000 | 1.0000 | 1.0000 | 0.3000 |
| wrong_dn | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| wrong_known_brand | 1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| wrong_material | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| wrong_pn | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Failure Stages

- false_exact_match: 1
- ld_mapping_failure: 8
- ok: 6
- parser_failure: 2
- retrieval_failure: 3

## Worst Cases

### parser_failure
- `natural_language_0004` `natural_language` `Нужен аналог ALSO DN500 PN16 фланцевое сталь 20` (status `exact_match`, invalid=2)
- `natural_language_0017` `natural_language` `Нужен аналог ALSO DN200 PN25 фланцевое сталь 20` (status `exact_match`, invalid=1)

### retrieval_failure
- `article_exact_0007` `article_exact` `1004718` (status `cannot_process`, invalid=0)
- `article_partial_0008` `article_partial` `100` (status `cannot_process`, invalid=0)
- `article_exact_0020` `article_exact` `1038452` (status `cannot_process`, invalid=0)

### false_exact_match
- `wrong_known_brand_0013` `wrong_known_brand` `Broen DN500 PN16` (status `exact_match`, invalid=5)

### ld_mapping_failure
- `brand_dn_pn_0001` `brand_dn_pn` `ALSO DN500 PN16` (status `exact_match`, invalid=1)
- `brand_dn_pn_compact_0002` `brand_dn_pn_compact` `ALSO DN500PN16` (status `exact_match`, invalid=2)
- `brand_dn_pn_material_0005` `brand_dn_pn_material` `ALSO DN500 PN16 сталь 20` (status `exact_match`, invalid=0)
- `brand_dn_pn_connection_0006` `brand_dn_pn_connection` `ALSO DN500 PN16 фланцевое` (status `exact_match`, invalid=2)
- `brand_dn_pn_0014` `brand_dn_pn` `ALSO DN200 PN25` (status `exact_match`, invalid=2)
- `brand_dn_pn_compact_0015` `brand_dn_pn_compact` `ALSO DN200PN25` (status `exact_match`, invalid=2)
- `brand_dn_pn_material_0018` `brand_dn_pn_material` `ALSO DN200 PN25 сталь 20` (status `exact_match`, invalid=0)
- `brand_dn_pn_connection_0019` `brand_dn_pn_connection` `ALSO DN200 PN25 фланцевое` (status `exact_match`, invalid=2)

CRITICAL: false_exact_match_rate > 0
CRITICAL: invalid_competitor_rate > 0
