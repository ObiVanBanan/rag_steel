# DeepSeek V5 Evaluation

## Overall
- cases: `180`
- brand interpretation accuracy: `1.0000`
- article identity accuracy: `0.9889`
- dn / pn / connection accuracy: `0.9840` / `0.9915` / `1.0000`
- hard interpretation exact match: `0.9722`
- hallucination rate: `0.0056`
- technical error rate: `0.0000`
- semantic correction success rate: `1.0000`
- ambiguous-case safety rate: `1.0000`
- latency p50 / p95: `1515.6` / `1778.2` ms

## Failures
- `v5_article_only_normalized_0048` `107 5450` wrong=[] missing=[] hallucinated=['dn'] error=-
- `v5_article_hard_conflict_0108` `107-5450 DN22 PN43 резьбовое` wrong=['dn'] missing=[] hallucinated=[] error=-
- `v5_article_hard_conflict_0109` `1163065 DN107 PN19 фланцевое` wrong=['dn'] missing=[] hallucinated=[] error=-
- `v5_v3_regression_0147` `MARSHAL DN40 PN40 фланцевое Цф.00.1.040.040` wrong=[] missing=['article'] hallucinated=[] error=-
- `v5_v3_regression_0153` `Temper DN107 PN999` wrong=[] missing=['pn_bar'] hallucinated=[] error=-
