# Model Comparison

- Dataset: `eval\queries.jsonl`
- Generated: `2026-08-09 11:16 UTC`

## Selection Order

Models are ranked by `LD nDCG@10`, then `LD Recall@10`, then `p95 latency`, then memory.

| Model | nDCG@10 | Recall@10 | MRR | Precision@10 | p50 ms | p95 ms | RAM MB | VRAM MB | Index points | Indexing ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 0.3768 | 0.4483 | 0.3463 | 0.1131 | 658.9 | 1041.7 | 306.5 | n/a | 16016 | 656949.2 |

## Selected Model

`text-embedding-3-small` ranks first by the plan's tie-break rules.

Reason:
- highest `LD nDCG@10` at 0.3768
- `LD Recall@10` at 0.4483
- `p95 latency` at 1041.7 ms
- peak RAM at 306.5 MiB

## Notes

- `RAM MB` is Python peak traced memory when available.
- `VRAM MB` is reported when CUDA is available.
- `Index points` uses the active Qdrant collection point count.
- `No-match FP rate` is reported only for queries with empty gold sets.
- `query_examples` in the JSON results stores per-query returned articles and top results.
