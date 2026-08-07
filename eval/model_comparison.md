# Model Comparison

- Dataset: `eval\queries.jsonl`
- Generated: `2026-08-06 14:00 UTC`

## Selection Order

Models are ranked by `LD nDCG@20`, then `LD Recall@20`, then `p95 latency`, then memory.

| Model | nDCG@20 | Recall@20 | MRR | Precision@20 | p50 ms | p95 ms | RAM MB | VRAM MB | Index points | Indexing ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BAAI/bge-m3 | 0.4483 | 0.5835 | 0.3899 | 0.0751 | 374.3 | 559.6 | 272.9 | 2725.5 | 16016 | 552709.3 |
| intfloat/multilingual-e5-base | 0.4146 | 0.5855 | 0.3414 | 0.0771 | 262.7 | 392.9 | 252.9 | 1495.9 | 16016 | 319696.9 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.3390 | 0.4825 | 0.2806 | 0.0646 | 223.3 | 344.8 | 433.8 | 593.1 | 16016 | 187240.6 |

## Selected Model

`BAAI/bge-m3` ranks first by the plan's tie-break rules.

Reason:
- highest `LD nDCG@20` at 0.4483
- `LD Recall@20` at 0.5835
- `p95 latency` at 559.6 ms
- peak RAM at 272.9 MiB

## Notes

- `RAM MB` is Python peak traced memory when available.
- `VRAM MB` is reported when CUDA is available.
- `Index points` uses the active Qdrant collection point count.
- `No-match FP rate` is reported only for queries with empty gold sets.
