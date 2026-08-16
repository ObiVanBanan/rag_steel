# RAG Steel

Hybrid search service for mapping third-party steel valve queries to LD products.

## What It Does

- Builds a versioned Qdrant index from `mapping_results.csv`
- Runs hybrid retrieval with:
  - dense embeddings
  - Qdrant BM25 sparse search
  - RRF fusion
  - source-to-LD expansion and LD deduplication
- Exposes a FastAPI API for search
- Includes evaluation scripts for offline quality checks

## Default Production Profile

Current dense embedding default:

```env
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
OPENAI_BASE_URL=https://api.openai.com/v1
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
DENSE_BATCH_SIZE=32
MAX_CONCURRENT_SEARCHES=8
QDRANT_TIMEOUT_SECONDS=5
UPSTREAM_MAX_ATTEMPTS=2
UPSTREAM_RETRY_BASE_DELAY_SECONDS=0.25
SOURCE_CANDIDATE_LIMIT=300
```

Notes:

- `text-embedding-3-small` is used only as the dense retriever.
- Sparse retrieval remains Qdrant BM25.
- Fusion remains RRF.
- Changing the embedding model or embedding dimension requires a full reindex.
- `OPENAI_API_KEY` must be present at runtime for OpenAI-based indexing and search.
- `DEEPSEEK_API_KEY` is required for structured v2 extraction when using the live DeepSeek path.
- Search requests are guarded by a small in-process concurrency gate.
- OpenAI and Qdrant calls use bounded retries and explicit request timeouts.

## Requirements

- Python `3.11`
- `uv`
- Docker
- Local Qdrant

## Install

```bash
uv sync
```

For local Hugging Face or SentenceTransformer embeddings, install the extra dependencies:

```bash
uv sync --extra local
```

## Docker Compose

The main runtime uses [compose.yaml](/C:/Users/theso/Desktop/job/rag_steel/compose.yaml:1) with:

- `qdrant`
- `api`
- `indexer` under the `tools` profile

Typical commands:

```bash
docker compose up -d qdrant
docker compose up -d api
docker compose --profile tools run --rm indexer
docker compose logs -f api
docker compose logs -f qdrant
docker compose down
```

Inside containers the API reaches Qdrant at `http://qdrant:6333`. From the host use `http://127.0.0.1:6333`.
If your deployment needs outbound proxy access, set `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` in `.env` instead of hardcoding them in `compose.yaml`.
The default container build installs only the OpenAI runtime path and does not include local GPU embedding dependencies.

## Deployment Notes

- Qdrant data lives in the named Docker volume from `compose.yaml`.
- Back up the volume by mounting it into a helper container and archiving `/qdrant/storage`.
- Keep proxy credentials and API keys in `.env` or deployment secrets, not in git.

## Build The Index

```bash
uv run python indexer.py --csv mapping_results.csv --recreate
```

The indexer:

- builds dense vectors
- stores sparse BM25 payload
- creates a versioned Qdrant collection
- switches the active alias only when `--recreate` is passed

## Load Test

Run the V2 load harness against the in-process API or a live deployment:

```bash
uv run python eval/load_test_v2.py
uv run python eval/load_test_v2.py --base-url http://127.0.0.1:8000
```

By default it exercises concurrency levels `1`, `5`, `10`, `20`, and `50`, then writes a markdown report and JSON results under `eval/`. Without `--base-url` it uses `httpx.ASGITransport` against the in-process app.

## Run The API

Local:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8005
```

Docker image:

- exposes port `8005`
- runs a single Uvicorn worker

Endpoints:

- `POST /v1/search`
- `POST /search`
- `POST /analogs`
- `GET /health/live`
- `GET /health/ready`

## Request Examples

`/v1/search`:

```bash
curl -X POST http://127.0.0.1:8005/v1/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Temper DN80 PN16\",\"limit\":20,\"include_debug\":false}"
```

Legacy wrapper:

```bash
curl -X POST http://127.0.0.1:8005/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Broen Ду80 Ру16\",\"top_k\":10,\"use_hybrid\":true}"
```

## Evaluation

Build the evaluation dataset if needed:

```bash
uv run python eval/build_eval_dataset.py
```

Run evaluation:

```bash
uv run python eval/evaluate.py
```

Current evaluation defaults:

- metrics are reported at `top_k=10`
- examples are saved for manual review

Artifacts:

- `eval/model_comparison.md`
- `eval/results/<run_id>.json`
- `eval/results/<run_id>_examples.json`

The examples file stores per-query:

- query text
- expected LD articles
- returned LD articles
- first relevant rank
- latency
- top returned results with scores

## Model Comparison

Default model set:

- `text-embedding-3-small`
- `paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-m3`

Ranking order:

1. `LD nDCG@10`
2. `LD Recall@10`
3. `p95 latency`
4. memory usage

## Search Result Scores

Each returned result exposes a real `score` from the hybrid search pipeline.
The API no longer depends on legacy explanation fields.

## Known Limits

- API and indexer depend on Qdrant availability.
- Dense indexes must be rebuilt per embedding model.
- Evaluation quality depends on the real model and real Qdrant state.
- The system uses one unified search pipeline and does not route queries by type.

## Key Files

- [main.py](/C:/Users/theso/Desktop/job/rag_steel/main.py:1)
- [indexer.py](/C:/Users/theso/Desktop/job/rag_steel/src/rag_steel/indexer.py:1)
- [search_engine.py](/C:/Users/theso/Desktop/job/rag_steel/src/rag_steel/search_engine.py:1)
- [settings.py](/C:/Users/theso/Desktop/job/rag_steel/src/rag_steel/settings.py:1)
- [embeddings.py](/C:/Users/theso/Desktop/job/rag_steel/src/rag_steel/embeddings.py:1)
- [eval/embeddings.py](/C:/Users/theso/Desktop/job/rag_steel/eval/embeddings.py:1)
- [data_builder.py](/C:/Users/theso/Desktop/job/rag_steel/src/rag_steel/data_builder.py:1)
- [.env.example](/C:/Users/theso/Desktop/job/rag_steel/.env.example:1)
- [compose.yaml](/C:/Users/theso/Desktop/job/rag_steel/compose.yaml:1)
- [Dockerfile](/C:/Users/theso/Desktop/job/rag_steel/Dockerfile:1)
