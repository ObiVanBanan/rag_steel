# Model Comparison

- Dataset: `eval/queries.jsonl`
- Generated: `2026-08-10 14:09 UTC`

## Selection Order

Models are ranked by `LD nDCG@10`, then `LD Recall@10`, then `p95 latency`, then memory.

| Model | nDCG@10 | Recall@10 | MRR | Precision@10 | p50 ms | p95 ms | RAM MB | VRAM MB | Index points | Indexing ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |
| intfloat/multilingual-e5-base | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |
| BAAI/bge-m3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0 | n/a | n/a | 0 | 0.0 |

## Selected Model

No models completed successfully.

## Failed Models

- `text-embedding-3-small`: HTTPStatusError: Client error '403 Forbidden' for url 'https://api.openai.com/v1/embeddings'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403
- `paraphrase-multilingual-MiniLM-L12-v2`: ModuleNotFoundError: No module named 'torch'
- `intfloat/multilingual-e5-base`: ModuleNotFoundError: No module named 'torch'
- `BAAI/bge-m3`: ModuleNotFoundError: No module named 'torch'
