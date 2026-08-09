# RAG Steel

## Production Embedding Profile

Current production dense embedding default:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_DEVICE=cpu
EMBEDDING_DTYPE=float32
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_SEQ_LENGTH=8191
OPENAI_BASE_URL=https://api.openai.com/v1
DENSE_BATCH_SIZE=32
SOURCE_CANDIDATE_LIMIT=300
```

Notes:

- `text-embedding-3-small` is used only as the dense retriever.
- Sparse retrieval remains Qdrant BM25.
- Fusion remains RRF.
- Query and document prefixes are empty for `text-embedding-3-small`.
- Changing the embedding model or embedding dimension requires a full reindex into a new Qdrant collection.
- `OPENAI_API_KEY` must be set at runtime for indexing and search.
- Keep the previous collection available for alias-based rollback.

## Docker Compose

The production-like container setup uses one `compose.yaml` with three services:

- `qdrant` for the vector database on CPU
- `api` for FastAPI plus `BAAI/bge-m3` on GPU
- `indexer` as an optional `tools` profile that reuses the same runtime image

Core commands:

```bash
docker compose up -d qdrant
docker compose up -d api
docker compose up -d
docker compose --profile tools run --rm indexer
docker compose logs -f api
docker compose logs -f qdrant
docker compose down
```

Inside containers the API must talk to Qdrant via `http://qdrant:6333`, while host-side checks can still use `http://127.0.0.1:6333`.

Поисковый сервис для подбора LD-аналогов по каталогу стальных изделий.

## Назначение

Проект решает задачу поиска LD-товаров по пользовательскому запросу и всегда возвращает именно LD-кандидаты, а не исходные steel-товары.

Пользовательский запрос проходит через единый hybrid pipeline:

1. нормализация запроса;
2. dense embedding;
3. sparse BM25;
4. объединение результатов через RRF;
5. rerank и дедупликация LD-кандидатов.

В runtime нет маршрутизации запросов по разным сценариям. Любой запрос идёт в один и тот же search flow.

## Как устроены данные

Исходный CSV `mapping_results.csv` группируется в source-product документы.

Для каждого source-product:

- собирается стабильный `steel_id`;
- строится `semantic_text` для dense embedding;
- строится `lexical_text` для BM25;
- собираются уникальные LD-кандидаты;
- сохраняются связи source -> LD.

Это важно, потому что поиск ранжирует исходные source-карточки, а затем уже из них выбирает LD-товары.

## Требования

- Python `3.11`
- `uv`
- `docker`
- локально запущенный Qdrant

## Установка

```bash
uv sync
```

Если нужны dev-зависимости, они уже описаны в `pyproject.toml` и ставятся через `uv sync`.

## Запуск Qdrant

```bash
docker compose up -d qdrant
```

По умолчанию локальный запуск ждёт Qdrant по адресу `http://localhost:6333`, а в Docker Compose адрес переопределяется на `http://qdrant:6333`.

## Подготовка индекса

Сначала можно профилировать CSV:

```bash
uv run python data_builder.py --csv data/mapping_results.csv
```

Затем построить индекс:

```bash
uv run python indexer.py --csv data/mapping_results.csv --recreate
```

Индексатор:

- строит dense-вектора;
- пишет sparse BM25 payload;
- создаёт versioned Qdrant collection;
- переключает alias только если указан `--recreate`.

## Запуск API

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Docker runtime uses the same application entrypoint but pins a single Uvicorn worker to avoid loading multiple GPU copies of `BAAI/bge-m3`.

Доступные endpoints:

- `POST /v1/search`
- `POST /search`
- `POST /analogs`
- `GET /health/live`
- `GET /health/ready`

## Примеры запросов

### v1 search

```bash
curl -X POST http://127.0.0.1:8000/v1/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Temper DN80 PN16\",\"limit\":20,\"include_debug\":false}"
```

### legacy wrapper

```bash
curl -X POST http://127.0.0.1:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Broen Ду80 Ру16\",\"top_k\":10,\"use_hybrid\":true}"
```

Если нужно подробное состояние нормализации запроса, поставьте `include_debug=true`.

## Тесты

```bash
uv run pytest
```

Линт:

```bash
uv run ruff check .
```

## Evaluation

Сначала собрать eval dataset, если это нужно:

```bash
uv run python eval/build_eval_dataset.py
```

Затем прогнать evaluation:

```bash
uv run python eval/evaluate.py
```

Результаты сохраняются в:

- `eval/model_comparison.md`
- `eval/results/<run_id>.json`

## Сравнение моделей

Для сравнения embedding-моделей используются:

- `paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-m3`

Отбор модели идёт по порядку:

1. `LD nDCG@20`
2. `LD Recall@20`
3. `p95 latency`
4. memory usage

Важно:

- для каждой модели dense-индекс пересобирается отдельно;
- sparse BM25 остаётся одинаковым;
- для E5 используются правильные префиксы `query:` и `passage:`;
- `BAAI/bge-m3` сравнивается как dense-модель, без использования её sparse-режимов.

## Как интерпретировать `relevance_rating`

`relevance_rating` в API это удобная шкала 0-100, полученная из итогового score.

На практике:

- чем выше `relevance_rating`, тем лучше результат;
- значение помогает быстро сравнивать кандидатов в ответе;
- для точной диагностики смотрите `score_breakdown`.

`score_breakdown` показывает, из чего собрался финальный score:

- `hybrid_score`
- `text_exactness`
- `source_score`
- `ld_field_score`
- `final_score`

## Известные ограничения

- API и индексатор зависят от доступности Qdrant.
- Embedding-модели скачиваются отдельно при первом запуске.
- Dense-индекс нужно пересобирать для каждой модели, иначе сравнение будет нечестным.
- `relevance_rating` не является вероятностью и не должен трактоваться как confidence.
- Benchmark для сравнения моделей требует реального окружения с моделями и Qdrant, поэтому локально может быть доступен только runner и тесты.
- Проект не делает query routing между разными пайплайнами, он всегда использует unified search.

## Полезные файлы

- [main.py](main.py)
- [indexer.py](indexer.py)
- [search_engine.py](search_engine.py)
- [data_builder.py](data_builder.py)
- [.env.example](.env.example)
- [compose.yaml](compose.yaml)
- [Dockerfile](Dockerfile)
- [.dockerignore](.dockerignore)
