# Live V2 Search Audit

## Scope

- generated_at: `2026-08-27T18:20:56.282546+00:00`
- base_url: `http://127.0.0.1:8000`
- endpoint: `POST /v2/search`
- suite: `smoke`
- dataset: `eval\data\live_v2_search_audit.jsonl`

## Health

- live: `{'http_status': 200, 'body': {'status': 'ok'}}`
- ready: `{'http_status': 200, 'body': {'status': 'ok', 'collection_alias': 'steel_products_active', 'resolved_collection_name': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'qdrant': {'alias': 'steel_products_active', 'resolved_collection': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'vector_dimension': 1536}, 'index': {'compatible': True, 'reason': None, 'schema_version': 'v2', 'index_format_version': 1, 'embedding_model': 'text-embedding-3-small', 'embedding_dimension': 1536, 'warnings': []}, 'details': {'runtime_model': 'text-embedding-3-small', 'runtime_revision': '', 'runtime_dimension': 1536, 'index_schema_version': 2, 'index_model': 'text-embedding-3-small', 'index_revision': '', 'index_dimension': 1536, 'qdrant_dense_vector_dimension': 1536, 'collection_alias': 'steel_products_active', 'resolved_collection_name': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'deepseek_configured': True, 'deepseek_model': 'deepseek-v4-flash'}}}`

## Summary

- total: `10`
- passed: `3`
- failed: `7`
- product_bugs: `6`
- data_missing: `0`
- presentation_issues: `0`
- checker_issues: `1`
- expected_normalization: `0`

## Metrics

- extraction_dn_accuracy: `0.2500`
- article_identity_accuracy: `1.0000`
- hard_constraint_accuracy: `unavailable`
- conflict_detection_accuracy: `0.0000`
- no_duplicate_result_rate: `1.0000`

## Latency

| Mode | p50 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: |
| article | 1650.9967 | 2611.8101 | 2611.8101 |
| semantic | 1411.6835 | 4529.1961 | 4529.1961 |

## Breakdown

| Category | Total | Classifications | DN Acc | Article Acc | Hard Acc | Conflict Acc | Dedup Rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| article_conflict | 1 | `{'PASS': 1}` | 1.0000 | unavailable | unavailable | unavailable | unavailable |
| article_identity | 1 | `{'EVAL_CHECKER_BUG': 1}` | unavailable | 1.0000 | unavailable | unavailable | 1.0000 |
| brand_conflict | 1 | `{'PRODUCT_BUG': 1}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| dn_boundary | 3 | `{'PRODUCT_BUG': 3}` | 0.0000 | unavailable | unavailable | unavailable | unavailable |
| dn_conflict | 1 | `{'PRODUCT_BUG': 1}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| pn_conflict | 1 | `{'PRODUCT_BUG': 1}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| short_numeric_article | 2 | `{'PASS': 2}` | unavailable | 1.0000 | unavailable | unavailable | 1.0000 |

## Findings

### EVAL_CHECKER_BUG - anchor_ball_dotted

- category: `article_identity`
- query: `КШ.Ф.П.Р.015.40-01`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### PRODUCT_BUG - dn_boundary_999

- category: `dn_boundary`
- query: `PALUR DN999 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 999 got None']`

### PRODUCT_BUG - dn_boundary_1000

- category: `dn_boundary`
- query: `PALUR DN1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['status expected exact_match got cannot_process', 'requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_boundary_1200

- category: `dn_boundary`
- query: `PALUR DN1200 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1200 got None']`

### PRODUCT_BUG - dn_conflict_temper

- category: `dn_conflict`
- query: `Temper DN50 DN80 PN16 фланцевое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_DN_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - pn_conflict_temper

- category: `pn_conflict`
- query: `Temper DN50 PN16 PN40 фланцевое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_PN_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - brand_conflict_temper_broen

- category: `brand_conflict`
- query: `Temper Broen DN50 PN16 фланцевое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_BRAND_CONFLICT_NOT_DETECTED']`

## Gaps

- This audit exercises the live API only; unit tests for the search architecture are separate.
- `expected_search_mode` is stored in the dataset, but the API response does not expose search mode yet, so the runner does not hard-fail on that field.
- Metrics with no applicable denominator are rendered as `unavailable`, not `0.0000`.
