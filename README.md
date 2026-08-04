# RAG Steel

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
docker compose up -d
```

По умолчанию сервис ждёт Qdrant по адресу `http://localhost:6333`.

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
- [docker-compose.yml](docker-compose.yml)
