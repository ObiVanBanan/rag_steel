# Live V2 Search Audit

## Scope

- generated_at: `2026-08-27T18:25:58.503676+00:00`
- base_url: `http://127.0.0.1:8000`
- endpoint: `POST /v2/search`
- suite: `full`
- dataset: `eval\data\live_v2_search_audit.jsonl`

## Health

- live: `{'http_status': 200, 'body': {'status': 'ok'}}`
- ready: `{'http_status': 200, 'body': {'status': 'ok', 'collection_alias': 'steel_products_active', 'resolved_collection_name': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'qdrant': {'alias': 'steel_products_active', 'resolved_collection': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'vector_dimension': 1536}, 'index': {'compatible': True, 'reason': None, 'schema_version': 'v2', 'index_format_version': 1, 'embedding_model': 'text-embedding-3-small', 'embedding_dimension': 1536, 'warnings': []}, 'details': {'runtime_model': 'text-embedding-3-small', 'runtime_revision': '', 'runtime_dimension': 1536, 'index_schema_version': 2, 'index_model': 'text-embedding-3-small', 'index_revision': '', 'index_dimension': 1536, 'qdrant_dense_vector_dimension': 1536, 'collection_alias': 'steel_products_active', 'resolved_collection_name': 'steel_products_text-embedding-3-small_20260823T180632Z', 'point_count': 19067, 'deepseek_configured': True, 'deepseek_model': 'deepseek-v4-flash'}}}`

## Summary

- total: `71`
- passed: `41`
- failed: `30`
- product_bugs: `23`
- data_missing: `1`
- presentation_issues: `0`
- checker_issues: `6`
- expected_normalization: `0`

## Metrics

- extraction_dn_accuracy: `0.2667`
- article_identity_accuracy: `1.0000`
- hard_constraint_accuracy: `1.0000`
- conflict_detection_accuracy: `0.0000`
- no_duplicate_result_rate: `1.0000`

## Latency

| Mode | p50 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: |
| article | 1873.6351 | 2642.5957 | 2642.5957 |
| semantic | 1672.7024 | 2183.5208 | 2592.7151 |

## Breakdown

| Category | Total | Classifications | DN Acc | Article Acc | Hard Acc | Conflict Acc | Dedup Rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| article_conflict | 5 | `{'PASS': 4, 'EVAL_CHECKER_BUG': 1}` | 1.0000 | 1.0000 | 1.0000 | unavailable | 1.0000 |
| article_identity | 5 | `{'EVAL_CHECKER_BUG': 5}` | unavailable | 1.0000 | unavailable | unavailable | 1.0000 |
| brand_conflict | 3 | `{'PRODUCT_BUG': 3}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| brand_normalization | 6 | `{'PASS': 6}` | unavailable | unavailable | 1.0000 | unavailable | 1.0000 |
| brass | 7 | `{'PASS': 7}` | unavailable | unavailable | 1.0000 | unavailable | 1.0000 |
| butterfly | 4 | `{'PASS': 3, 'PRODUCT_BUG': 1}` | unavailable | unavailable | 1.0000 | 0.0000 | 1.0000 |
| connection_conflict | 4 | `{'PRODUCT_BUG': 4}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| dn_boundary | 5 | `{'PASS': 1, 'PRODUCT_BUG': 4}` | 0.2000 | unavailable | 1.0000 | unavailable | 1.0000 |
| dn_conflict | 2 | `{'PRODUCT_BUG': 2}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| extraction | 6 | `{'PRODUCT_BUG': 6}` | 0.0000 | unavailable | unavailable | unavailable | unavailable |
| material_conflict | 1 | `{'PRODUCT_BUG': 1}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| pn_conflict | 1 | `{'PRODUCT_BUG': 1}` | unavailable | unavailable | unavailable | 0.0000 | 1.0000 |
| pn_semantics | 5 | `{'PASS': 5}` | unavailable | unavailable | 1.0000 | unavailable | 1.0000 |
| production_anchor | 1 | `{'DATA_MISSING': 1}` | unavailable | unavailable | unavailable | unavailable | unavailable |
| result_integrity | 3 | `{'PASS': 2, 'PRODUCT_BUG': 1}` | 0.0000 | unavailable | 1.0000 | unavailable | 1.0000 |
| short_numeric_article | 5 | `{'PASS': 5}` | unavailable | 1.0000 | unavailable | unavailable | 1.0000 |
| short_numeric_negative | 5 | `{'PASS': 5}` | unavailable | unavailable | unavailable | unavailable | unavailable |
| steel | 1 | `{'PASS': 1}` | unavailable | unavailable | 1.0000 | unavailable | 1.0000 |
| unknown_article | 2 | `{'PASS': 2}` | unavailable | unavailable | unavailable | unavailable | unavailable |

## Findings

### EVAL_CHECKER_BUG - anchor_ball_dotted

- category: `article_identity`
- query: `КШ.Ф.П.Р.015.40-01`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': 'ручное', 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### EVAL_CHECKER_BUG - anchor_ball_dotted_nl

- category: `article_identity`
- query: `Нужен аналог КШ.Ф.П.Р.015.40-01`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### EVAL_CHECKER_BUG - anchor_ball_compact

- category: `article_identity`
- query: `КШ.ФПР.015.40-01`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.ФПР.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### EVAL_CHECKER_BUG - anchor_ball_label

- category: `article_identity`
- query: `Артикул: КШ.Ф.П.Р.015.40-01`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': 'ручное', 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### EVAL_CHECKER_BUG - anchor_ball_quoted

- category: `article_identity`
- query: `"КШ.Ф.П.Р.015.40-01"`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': None, 'pn_bar': None, 'connection': None, 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### DATA_MISSING - anchor_butterfly_missing

- category: `production_anchor`
- query: `ТМ.3.03.03.01.200.16.С/С`
- status: `not_found`
- reason: `{'code': 'ARTICLE_NOT_FOUND', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': None, 'article': 'ТМ.3.03.03.01.200.16.С/С', 'dn': 200.0, 'pn_bar': 16.0, 'connection': 'сварное', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `[]`

### EVAL_CHECKER_BUG - anchor_ball_compatible

- category: `article_conflict`
- query: `Нужен аналог КШ.Ф.П.Р.015.40-01 DN15 PN40`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'ALSO', 'article': 'КШ.Ф.П.Р.015.40-01', 'dn': 15.0, 'pn_bar': 40.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None, 'resolved_article': 'КШ.ФПР.015.40-01'}`
- issues: `['raw article alias differs: expected КШ.Ф.П.Р.015.40-01 got КШ.ФПР.015.40-01']`

### PRODUCT_BUG - dn_boundary_900

- category: `dn_boundary`
- query: `PALUR DN900 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 900 got None']`

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

### PRODUCT_BUG - dn_syntax_1000_compact

- category: `extraction`
- query: `PALUR DN1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_syntax_1000_space

- category: `extraction`
- query: `PALUR DN 1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_syntax_1000_lower

- category: `extraction`
- query: `PALUR dn1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_syntax_1000_ru_compact

- category: `extraction`
- query: `PALUR Ду1000 Ру16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_syntax_1000_ru_space

- category: `extraction`
- query: `PALUR Ду 1000 Ру 16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - dn_syntax_1000_ru_upper

- category: `extraction`
- query: `PALUR ДУ1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

### PRODUCT_BUG - connection_conflict_temper_plain

- category: `connection_conflict`
- query: `Temper DN50 PN16 фланцевое резьбовое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_CONNECTION_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - connection_conflict_temper_and

- category: `connection_conflict`
- query: `Temper DN50 PN16 фланцевое и резьбовое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_CONNECTION_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - connection_conflict_temper_welded

- category: `connection_conflict`
- query: `Temper DN50 PN16 сварное фланцевое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Temper', 'article': None, 'dn': 50.0, 'pn_bar': 16.0, 'connection': 'сварное', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_CONNECTION_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - connection_conflict_palur

- category: `connection_conflict`
- query: `PALUR DN200 PN16 фланцевое резьбовое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': 200.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_CONNECTION_CONFLICT_NOT_DETECTED']`

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

### PRODUCT_BUG - brand_conflict_valtec_stout

- category: `brand_conflict`
- query: `Valtec Stout DN20 PN40 резьбовое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Stout', 'article': None, 'dn': 20.0, 'pn_bar': 40.0, 'connection': 'резьбовое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_BRAND_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - material_conflict_valtec

- category: `material_conflict`
- query: `Valtec DN20 PN40 латунный стальной резьбовое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'Valtec', 'article': None, 'dn': 20.0, 'pn_bar': 40.0, 'connection': 'резьбовое', 'body_material': 'латунь', 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_MATERIAL_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - dn_conflict_palur

- category: `dn_conflict`
- query: `PALUR DN200 DN300 PN16 фланцевое`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': 200.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_DN_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - brand_conflict_palur_temper

- category: `brand_conflict`
- query: `PALUR Temper DN200 PN16`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': 200.0, 'pn_bar': 16.0, 'connection': None, 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_BRAND_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - butterfly_control_conflict

- category: `butterfly`
- query: `PALUR DN200 PN16 фланцевое ручное электропривод`
- status: `exact_match`
- reason: `None`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': 200.0, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': 'ручное', 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `["status expected ['cannot_process', 'not_found'] got exact_match", 'BUG_CONTROL_CONFLICT_NOT_DETECTED']`

### PRODUCT_BUG - semantic_palur_1000

- category: `result_integrity`
- query: `PALUR DN1000 PN16 фланцевое`
- status: `cannot_process`
- reason: `{'code': 'HARD_CONSTRAINT_UNRESOLVED', 'message': 'Подходящие товары не найдены. Возможен поиск по следующим брендам: Temper, ALSO, MARSHAL, Broen, ADL, FORTECA, Бивал, PALUR, Gallop, Aquasfera, БАЗ, Stout, STI, Valtec', 'retryable': False}`
- requested: `{'brand': 'PALUR', 'article': None, 'dn': None, 'pn_bar': 16.0, 'connection': 'фланцевое', 'body_material': None, 'medium': None, 'control': None, 'temperature': None, 'length_mm': None, 'series': None}`
- issues: `['requested.dn expected 1000 got None']`

## Gaps

- This audit exercises the live API only; unit tests for the search architecture are separate.
- `expected_search_mode` is stored in the dataset, but the API response does not expose search mode yet, so the runner does not hard-fail on that field.
- Metrics with no applicable denominator are rendered as `unavailable`, not `0.0000`.
