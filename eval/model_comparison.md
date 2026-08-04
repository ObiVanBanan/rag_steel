# Model Comparison

- Dataset: `eval/queries.jsonl`
- Status: Phase 13A benchmark runner is implemented and tested.
- Status: Phase 13B live benchmark and model selection are still pending.

## Models

- `paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-m3`

## Metrics

The evaluator captures:

- `LD Recall@20`
- `LD nDCG@20`
- `MRR`
- indexing time
- query latency p50
- query latency p95
- RAM usage
- VRAM usage
- index size

## Selection Order

Models are ranked by:

1. `LD nDCG@20`
2. `LD Recall@20`
3. `p95 latency`
4. memory usage

## Machine-Readable Output

The runner writes JSON results to `eval/results/<run_id>.json` and renders this Markdown report from the same run data.

## How To Run

When embedding weights and Qdrant are available, run:

```bash
python eval/evaluate.py --models paraphrase-multilingual-MiniLM-L12-v2 intfloat/multilingual-e5-base BAAI/bge-m3
```

The command will rebuild dense vectors for each model, evaluate the unified LD dataset, and overwrite this report with the measured results.
