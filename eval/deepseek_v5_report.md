# DeepSeek V5 Evaluation

## Overall
- cases: `180`
- brand interpretation accuracy: `0.0000`
- article identity accuracy: `0.0000`
- dn / pn / connection accuracy: `0.0000` / `0.0000` / `0.0000`
- hard interpretation exact match: `0.0000`
- hallucination rate: `0.0000`
- technical error rate: `1.0000`
- semantic correction success rate: `0.0000`
- ambiguous-case safety rate: `0.0000`
- latency p50 / p95: `26.0` / `37.2` ms

## Failures
- `v5_brand_semantic_0001` `Броен DN50` wrong=[] missing=['brand', 'dn'] hallucinated=[] error=upstream_error
- `v5_brand_semantic_0002` `Маршал сотка` wrong=[] missing=['brand', 'dn'] hallucinated=[] error=upstream_error
- `v5_brand_semantic_0003` `Темпер DN50` wrong=[] missing=['brand', 'dn'] hallucinated=[] error=upstream_error
- `v5_brand_semantic_0004` `Темпр DN50` wrong=[] missing=['brand', 'dn'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0005` `ALSOO DN200 PN25 фланцевое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0006` `ALSOO DN500 PN16 фланцевое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0007` `Broem DN100 PN16 сварное` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0008` `Broem DN15 PN40 резьбовое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0009` `Fortecaa DN15 PN40 фланцевое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0010` `Marsha DN15 PN16 сварное` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0011` `Marsha DN20 PN16 сварное` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0012` `Tempr DN15 PN40 резьбовое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0013` `Tempr DN20 PN40 резьбовое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_brand_typo_0014` `Бивл DN25 PN40 фланцевое` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_mixed_semantic_0015` `нужен броен сотка на 25 под сварку` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_mixed_semantic_0016` `темпер ду51 ру16 фланец` wrong=[] missing=['brand', 'dn', 'pn_bar', 'connection'] hallucinated=[] error=upstream_error
- `v5_dn_semantic_0017` `DN51` wrong=[] missing=['dn'] hallucinated=[] error=upstream_error
- `v5_dn_semantic_0018` `ду 64` wrong=[] missing=['dn'] hallucinated=[] error=upstream_error
- `v5_dn_semantic_0019` `пятидесятый` wrong=[] missing=['dn'] hallucinated=[] error=upstream_error
- `v5_dn_semantic_0020` `сотка` wrong=[] missing=['dn'] hallucinated=[] error=upstream_error
