# Model Comparison

- Dataset: `eval\queries.jsonl`
- Generated: `2026-08-09 18:01 UTC`

## Selection Order

Models are ranked by `LD nDCG@10`, then `LD Recall@10`, then `p95 latency`, then memory.

| Model | nDCG@10 | Recall@10 | MRR | Precision@10 | p50 ms | p95 ms | RAM MB | VRAM MB | Index points | Indexing ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 0.3547 | 0.4348 | 0.3175 | 0.1082 | 466.8 | 575.1 | 306.5 | n/a | 16016 | 706035.7 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |
| intfloat/multilingual-e5-base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |
| BAAI/bge-m3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |

## Selected Model

`text-embedding-3-small` ranks first by the plan's tie-break rules.

Reason:
- highest `LD nDCG@10` at 0.3547
- `LD Recall@10` at 0.4348
- `p95 latency` at 575.1 ms
- peak RAM at 306.5 MiB

## Notes

- `RAM MB` is Python peak traced memory when available.
- `VRAM MB` is reported when CUDA is available.
- `Index points` uses the active Qdrant collection point count.
- `No-match FP rate` is reported only for queries with empty gold sets.
- `query_examples` in the JSON results stores per-query returned articles and top results.

## Failed Models

- `paraphrase-multilingual-MiniLM-L12-v2`: OSError: [Errno 22] Invalid argument
- `intfloat/multilingual-e5-base`: OSError: [Errno 22] Invalid argument
- `BAAI/bge-m3`: OSError: [Errno 22] Invalid argument
