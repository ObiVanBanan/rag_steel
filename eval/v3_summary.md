# V3 Summary

Generated at `2026-08-17T20:20:28`

## Core
- DeepSeek hard exact: `0.9663`
- DeepSeek brand accuracy: `0.9667`
- RAG preferred hit@1: `0.9318`
- RAG preferred hit@5: `0.9886`
- RAG MRR: `0.9521`
- E2E preferred hit@1: `0.8977`
- E2E preferred hit@5: `0.9545`
- E2E overall pass: `0.9667`
- E2E strict overall pass: `0.9556`

## Dataset
- DeepSeek cases: `90`
- RAG cases: `90`
- E2E cases: `90`

## Latency
- DeepSeek p95: `1824.4` ms
- RAG qdrant p95: `175.6` ms
- E2E wall clock p95: `1860.5` ms

## Failure Stages
- DeepSeek: {'attribute_mismatch': 2, 'brand_gate_failure': 3}
- RAG: {'ok': 90}
- E2E: {'ok': 86, 'ranking_failure': 1, 'brand_gate_failure': 3}
