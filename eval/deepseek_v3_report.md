# DeepSeek V3 Evaluation

## Run
- commit: `3baad66`
- dataset: `eval\v3_golden_queries.jsonl`
- cases: `90`
- model: `deepseek-v4-flash`

## Overall
- brand accuracy: `0.9667`
- brand false positive rate: `0.0000`
- brand false negative rate: `0.0337`
- extraction cases: `89`
- hard exact match rate: `0.9663`
- dn / pn / connection accuracy: `0.9634` / `0.9634` / `0.9846`
- hard hallucination rate: `0.0000`
- soft hallucination rate: `0.0112`
- hard missing rate: `0.0337`
- soft missing rate: `0.0112`
- invalid / timeout / upstream / config / unexpected: `0.0000` / `0.0000` / `0.0000` / `0.0000` / `0.0000`
- latency p50 / p95 / p99: `1529.9` / `1824.4` / `11503.5` ms

## By Category

| Category | Cases | Brand Acc | Hard Exact | Invalid |
| --- | ---: | ---: | ---: | ---: |
| compact_syntax | 7 | 0.8571 | 0.8571 | 0.0000 |
| hard_only | 7 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_article | 16 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_material | 7 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_medium | 7 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_multiple_soft | 7 | 1.0000 | 1.0000 | 0.0000 |
| hard_plus_series | 2 | 1.0000 | 1.0000 | 0.0000 |
| impossible_hard | 1 | 1.0000 | 0.0000 | 0.0000 |
| missing_connection | 7 | 1.0000 | 1.0000 | 0.0000 |
| missing_dn | 7 | 1.0000 | 0.8571 | 0.0000 |
| missing_pn | 7 | 1.0000 | 1.0000 | 0.0000 |
| natural_language | 7 | 0.8571 | 0.8571 | 0.0000 |
| no_brand | 1 | 1.0000 | 0.0000 | 0.0000 |
| russian_alias | 7 | 0.8571 | 0.8571 | 0.0000 |

## Worst Failures

- `v3_missing_dn_0049` `FORTECA PN25 фланцевое` wrong=[] missing=[] hallucinated=['series'] error=-
- `v3_russian_alias_0068` `also Ду50 Ру40` wrong=[] missing=['brand', 'dn', 'pn_bar'] hallucinated=[] error=-
- `v3_compact_syntax_0075` `ALSO DN50PN40` wrong=[] missing=['brand', 'dn', 'pn_bar'] hallucinated=[] error=-
- `v3_natural_language_0082` `Подбери мне аналог ALSO на диаметр 50 давление 40 бар штуцерное сталь 09г2с жидкость` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection', 'body_material', 'medium'] hallucinated=[] error=-
- `v3_no_brand_0089` `Нужен шаровой кран DN20 PN40 резьбовое` wrong=[] missing=['dn', 'pn_bar', 'connection'] hallucinated=[] error=-
