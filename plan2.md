# Перевод `rag_steel` на BAAI/bge-m3 и production-тестирование

## Цель

Сделать `BAAI/bge-m3` основной embedding-моделью проекта.

Сохранить существующую архитектуру:

```text
BGE-M3 dense embeddings
        +
Qdrant BM25 sparse search
        ↓
RRF fusion
        ↓
source reranking
        ↓
LD candidate ranking
        ↓
уникальные top-20 LD
```

Не добавлять в этой задаче:

* BGE-M3 sparse embeddings;
* ColBERT/multi-vector режим BGE-M3;
* новый reranker;
* query router;
* новую формулу рейтинга;
* ONNX;
* внешнюю embedding-службу;
* изменение evaluation dataset.

Меняется только production dense-модель и связанные с ней параметры.

---

# 1. Проверить текущее состояние

Перед изменениями выполнить:

```powershell
git status
git log --oneline -10
uv run pytest -q
uv run ruff check .
```

Зафиксировать текущие значения:

```text
embedding model
embedding dimension
collection alias
dense vector name
sparse vector name
batch size
max sequence length
device
dtype
```

Не продолжать, если тесты не проходят до изменений.

Создать ветку:

```powershell
git checkout -b feature/production-bge-m3
```

---

# 2. Обновить production-конфигурацию

Найти единственный source of truth для настроек.

Предпочтительный файл:

```text
src/rag_steel/config.py
```

Корневой `config.py`, если существует, должен быть только compatibility wrapper.

Не создавать второй независимый реестр моделей.

## 2.1. Рекомендуемый `.env.production`

Добавить или изменить:

```env
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_REVISION=
EMBEDDING_DIMENSION=1024

EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_SEQ_LENGTH=512

DENSE_BATCH_SIZE=32

QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_COLLECTION_ALIAS=steel_products_active

SOURCE_CANDIDATE_LIMIT=300
RESULT_LIMIT_DEFAULT=20
RESULT_LIMIT_MAX=100
```

Не добавлять BGE-префиксы:

```env
EMBEDDING_QUERY_PREFIX=
EMBEDDING_DOCUMENT_PREFIX=
```

BGE-M3 не требует:

```text
query:
passage:
Represent this sentence...
```

Префиксы E5 должны оставаться только в конфигурации E5.

## 2.2. Параметры production baseline

Использовать:

```text
model:                BAAI/bge-m3
dimension:            1024
distance:             Cosine
device:               CUDA
dtype:                float16
normalize embeddings: true
max sequence length:  512
index batch size:     32
source candidates:    300
```

Не использовать максимальную длину 8192 для коротких карточек товаров.

---

# 3. Закрепить revision модели

Не оставлять `main` в окончательной production-конфигурации.

Для первого скачивания разрешено временно использовать:

```env
EMBEDDING_REVISION=main
```

После скачивания определить текущий commit модели:

```powershell
uv run python -c "from huggingface_hub import model_info; print(model_info('BAAI/bge-m3').sha)"
```

Полученный SHA записать в `.env.production`:

```env
EMBEDDING_REVISION=<полный SHA модели>
```

Не придумывать SHA и не копировать случайный commit из интернета.

Добавить resolved revision в метаданные индекса:

```json
{
  "embedding_model": "BAAI/bge-m3",
  "embedding_revision": "<resolved SHA>",
  "embedding_dimension": 1024,
  "embedding_dtype": "float16",
  "max_sequence_length": 512
}
```

---

# 4. Обновить реестр embedding-моделей

В существующем model registry должна быть запись:

```python
EmbeddingModelSpec(
    model_id="BAAI/bge-m3",
    dimension=1024,
    query_prefix="",
    document_prefix="",
    normalize_embeddings=True,
    max_sequence_length=512,
    preferred_dtype="float16",
)
```

Не удалять MiniLM и E5 из benchmark registry.

Production default должен указывать на BGE-M3.

Пример:

```python
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
```

Реестр должен использоваться одновременно:

* индексатором;
* runtime search;
* benchmark runner;
* healthcheck;
* metadata validation.

Не разрешать разным компонентам самостоятельно определять размерность модели.

---

# 5. Исправить загрузку модели

Использовать `SentenceTransformer`.

Пример реализации:

```python
from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_embedding_model(settings) -> SentenceTransformer:
    if settings.embedding_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported embedding dtype: {settings.embedding_dtype}"
        )

    model = SentenceTransformer(
        settings.embedding_model,
        revision=settings.embedding_revision or None,
        device=settings.embedding_device,
        model_kwargs={
            "torch_dtype": DTYPE_MAP[settings.embedding_dtype],
        },
    )

    model.max_seq_length = settings.embedding_max_seq_length

    actual_dimension = model.get_sentence_embedding_dimension()

    if actual_dimension != settings.embedding_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"configured={settings.embedding_dimension}, "
            f"actual={actual_dimension}"
        )

    return model
```

Не вызывать:

```python
model.half()
```

если dtype уже задан через `model_kwargs`.

Не загружать модель на уровне импорта модуля.

Модель должна загружаться через:

```text
FastAPI lifespan
dependency container
application startup
```

Для GPU `float16` является нормальным первым production-профилем: Sentence Transformers поддерживает загрузку через `torch_dtype`, а fp16 обычно ускоряет GPU inference при небольшом влиянии на точность.

---

# 6. Кодирование документов и запросов

Для документов использовать:

```python
document_embeddings = model.encode_document(
    semantic_texts,
    batch_size=settings.dense_batch_size,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
```

Для запроса:

```python
query_embedding = model.encode_query(
    processed_query.semantic_text,
    batch_size=1,
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
)
```

Если установленная версия Sentence Transformers не поддерживает эти методы, разрешено использовать:

```python
model.encode(...)
```

Но параметры должны оставаться одинаковыми:

```python
normalize_embeddings=True
```

Для BGE-M3 не добавлять вручную `query:` и `passage:`.

`encode_query()` и `encode_document()` рекомендуются Sentence Transformers для retrieval-задач; у моделей без отдельных query/document prompts они работают эквивалентно обычному `encode()`.

---

# 7. Обновить создание Qdrant-коллекции

Старую коллекцию нельзя использовать повторно, если она создана под размерность 384 или 768.

BGE-M3 требует:

```text
dense vector size = 1024
```

Qdrant фиксирует размерность vector space при создании коллекции, поэтому переход на BGE-M3 требует полной переиндексации.

Новая коллекция:

```text
steel_products_bge_m3_<timestamp>
```

Конфигурация:

```python
vectors_config={
    settings.qdrant_dense_vector_name: models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    ),
},
sparse_vectors_config={
    settings.qdrant_sparse_vector_name: models.SparseVectorParams(),
},
```

Не менять sparse/BM25-конфигурацию.

Не удалять текущую active collection.

Порядок:

```text
1. Создать новую BGE-M3 коллекцию.
2. Загрузить все points.
3. Проверить размерность.
4. Проверить количество points.
5. Выполнить smoke queries.
6. Выполнить eval.
7. Только затем переключить alias.
8. Старую коллекцию сохранить для rollback.
```

---

# 8. Проверить индексатор

Индексатор должен вывести:

```text
model_id
model_revision
device
dtype
dimension
max_sequence_length
batch_size
source rows
grouped source products
Qdrant points
elapsed time
collection name
```

Добавить проверки:

```python
assert embeddings.ndim == 2
assert embeddings.shape[1] == 1024
assert len(embeddings) == len(documents)
```

Перед upsert проверить:

```python
if not np.isfinite(embeddings).all():
    raise RuntimeError("Embeddings contain NaN or infinity")
```

После индексации:

```text
expected points == actual Qdrant points
```

По предыдущему benchmark ожидается приблизительно:

```text
16016 points
```

Но не использовать это как постоянную константу.

---

# 9. Не менять hybrid search

Сохранить:

```text
dense BGE-M3
BM25
RRF
```

Не делать weighted sum сырых dense- и BM25-score.

Dense cosine и BM25 имеют разные шкалы, поэтому RRF остаётся безопасным способом fusion. Qdrant рекомендует выполнять retriever-запросы в `prefetch`, а fusion — в основном Query API query.

Сохранить существующую конструкцию:

```python
prefetch=[
    models.Prefetch(
        query=query_embedding.tolist(),
        using="dense",
        limit=settings.source_candidate_limit,
    ),
    models.Prefetch(
        query=models.Document(
            text=processed_query.lexical_text,
            model="qdrant/bm25",
            options={"tokenizer": "multilingual"},
        ),
        using="sparse",
        limit=settings.source_candidate_limit,
    ),
],
query=models.FusionQuery(
    fusion=models.Fusion.RRF,
),
```

Адаптировать точные параметры только под установленную версию `qdrant-client`.

Не менять одновременно:

```text
RRF
source scoring weights
LD scoring weights
candidate limit
normalization
BM25 tokenizer
```

---

# 10. Добавить healthcheck BGE-M3

`/health/ready` должен дополнительно проверять:

```text
configured model == index metadata model
configured revision == index metadata revision
configured dimension == 1024
Qdrant dense vector dimension == 1024
active alias существует
collection point count > 0
```

Если приложение загружено с BGE-M3, а alias указывает на E5/MiniLM collection:

```text
ready = false
```

Ошибка:

```json
{
  "status": "not_ready",
  "reason": "EMBEDDING_INDEX_MISMATCH",
  "details": {
    "runtime_model": "BAAI/bge-m3",
    "index_model": "intfloat/multilingual-e5-base"
  }
}
```

Не разрешать приложению молча выполнять поиск в несовместимом индексе.

---

# 11. Добавить unit-тесты конфигурации

Добавить проверки:

```text
test_bge_m3_is_production_default
test_bge_m3_dimension_is_1024
test_bge_m3_has_empty_query_prefix
test_bge_m3_has_empty_document_prefix
test_bge_m3_uses_normalization
test_bge_m3_uses_fp16_on_cuda
test_model_dimension_is_validated_after_loading
test_index_metadata_must_match_runtime_model
test_ready_fails_for_old_embedding_collection
```

Тесты не должны скачивать настоящую модель.

Использовать fake model:

```python
class FakeEmbeddingModel:
    max_seq_length = 512

    def get_sentence_embedding_dimension(self) -> int:
        return 1024
```

---

# 12. Перестроить production-кандидат индекса

Сначала скачать модель отдельно:

```powershell
uv run python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3', device='cuda', model_kwargs={'torch_dtype':'float16'}); print(m.get_sentence_embedding_dimension())"
```

Ожидается:

```text
1024
```

Затем запустить Qdrant:

```powershell
docker compose up -d qdrant
```

Проверить:

```powershell
docker compose ps
```

Запустить индексатор существующей командой проекта.

Перед запуском посмотреть поддерживаемые аргументы:

```powershell
uv run python indexer.py --help
```

Затем выполнить полную переиндексацию в новую коллекцию.

Не указывать несуществующие CLI-аргументы.

---

# 13. Production smoke test

После индексации запустить API:

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Проверить:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

## Обязательные запросы

### Точный артикул

```json
{
  "query": "1184399",
  "limit": 20
}
```

### Нормализованный технический артикул

```json
{
  "query": "кшпп0154001",
  "limit": 20
}
```

### Описание

```json
{
  "query": "Temper DN80 PN16",
  "limit": 20
}
```

### Смешанный запрос

```json
{
  "query": "Temper 1184399 Ду80 Ру16",
  "limit": 20
}
```

### Естественный язык

```json
{
  "query": "нужен фланцевый шаровой кран Temper Ду80 Ру16",
  "limit": 20
}
```

### Несуществующий товар

```json
{
  "query": "несуществующий кран XYZ-999999 DN777",
  "limit": 20
}
```

## Для каждого ответа проверить

```text
HTTP 200
results содержит только LD
ld_article не повторяется
count <= 20
rank идёт от 1 без пропусков
relevance_rating находится в диапазоне 0..100
нет NaN
есть source_evidence
нет нарушения явно указанного DN
```

---

# 14. Запустить полный evaluation

Использовать существующий runner.

Сначала:

```powershell
uv run python eval/evaluate.py --help
```

Затем запустить только BGE-M3 на полном:

```text
eval/queries.jsonl
```

Не придумывать CLI-параметры: использовать реально существующие параметры runner.

## Regression gates

Результат BGE-M3 не должен быть хуже предыдущего benchmark более чем на допустимое округление:

```text
LD nDCG@20    >= 0.44
LD Recall@20  >= 0.57
MRR           >= 0.38
Unique LD     == 1.00
p95 latency   <= 700 ms
```

Дополнительно вывести:

```text
nDCG@5
Precision@5
Recall@5
no-match false-positive rate
DN violation rate
PN violation rate
duplicate LD rate
```

Для технических нарушений:

```text
duplicate LD rate == 0
DN violation rate == 0
```

Не переключать production alias, если quality gates не пройдены.

---

# 15. Проверить нагрузку

Создать или использовать существующий load-test script на `httpx`.

Профили:

```text
1 concurrent request
4 concurrent requests
8 concurrent requests
```

На каждом уровне выполнить минимум 100 запросов из eval dataset.

Сохранить:

```text
requests
success rate
p50
p95
p99
requests per second
GPU peak memory
process RSS
Qdrant errors
HTTP 5xx
```

Production gates первой версии:

```text
HTTP success rate == 100%
HTTP 5xx == 0
p95 при concurrency 1 <= 700 ms
p95 при concurrency 4 <= 1500 ms
VRAM peak <= 4 GiB
```

Не использовать `tracemalloc` как реальное измерение RAM.

Для RAM использовать:

```python
psutil.Process().memory_info().rss
```

Для GPU:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Перед чтением GPU-метрик:

```python
torch.cuda.synchronize()
```

---

# 16. Протестировать три production-профиля

Сначала выполнить профиль A.

## Профиль A — рекомендуемый baseline

```env
EMBEDDING_DTYPE=float16
EMBEDDING_MAX_SEQ_LENGTH=512
DENSE_BATCH_SIZE=32
SOURCE_CANDIDATE_LIMIT=300
```

Это основной кандидат.

## Профиль B — низкая задержка

```env
EMBEDDING_DTYPE=float16
EMBEDDING_MAX_SEQ_LENGTH=256
DENSE_BATCH_SIZE=32
SOURCE_CANDIDATE_LIMIT=200
```

Проверить:

```text
насколько уменьшился p95
не упал ли nDCG@20 более чем на 0.01
не упал ли Recall@20 более чем на 0.01
```

Для смены `max_sequence_length` требуется переиндексация.

## Профиль C — повышенный recall

```env
EMBEDDING_DTYPE=float16
EMBEDDING_MAX_SEQ_LENGTH=512
DENSE_BATCH_SIZE=32
SOURCE_CANDIDATE_LIMIT=500
```

Переиндексация не требуется, если меняется только candidate limit.

Проверить:

```text
растёт ли Recall@20
растёт ли nDCG@20
насколько ухудшается p95
```

## Правило выбора

Выбрать профиль с максимальным `nDCG@20`, если:

```text
p95 <= 700 ms для одного запроса
VRAM <= 4 GiB
Recall@20 >= 0.57
DN violations == 0
duplicate LD rate == 0
```

Не выбирать профиль только по скорости.

---

# 17. Опциональная оптимизация после baseline

Не выполнять в первом коммите.

После успешного production baseline отдельно проверить:

```text
batch size: 16, 32, 64
max sequence length: 256, 512
attention: SDPA, flash_attention_2
dtype: fp16, bf16
```

Каждое изменение выполнять отдельным benchmark.

Flash Attention не добавлять автоматически.

Использовать только если:

```text
GPU поддерживается
зависимости устанавливаются воспроизводимо
полный test suite проходит
quality не ухудшается
p95 действительно уменьшается
```

Sentence Transformers поддерживает fp16, bf16 и Flash Attention через `model_kwargs`, однако эти оптимизации нужно измерять на конкретном GPU.

---

# 18. Переключить alias

Только после успешных:

```text
unit tests
full regression tests
evaluation
smoke test
load test
metadata validation
```

выполнить атомарное переключение:

```text
steel_products_active
old collection → new BGE-M3 collection
```

Старую collection оставить минимум до завершения ручной проверки.

Записать:

```text
old collection
new collection
alias
switch timestamp
model revision
CSV SHA256
eval result file
```

## Rollback

Rollback должен состоять только из обратного переключения alias.

Не требовать повторной индексации.

Проверить rollback до удаления старой collection:

```text
BGE-M3 → старая коллекция → BGE-M3
```

---

# 19. Обновить README

Добавить production-конфигурацию:

```env
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_NORMALIZE=true
EMBEDDING_MAX_SEQ_LENGTH=512
DENSE_BATCH_SIZE=32
SOURCE_CANDIDATE_LIMIT=300
```

Объяснить:

* BGE-M3 выбрана по максимальному `LD nDCG@20`;
* размерность равна 1024;
* после смены embedding-модели необходима переиндексация;
* BGE-M3 используется только в dense-режиме;
* sparse retrieval остаётся Qdrant BM25;
* fusion остаётся RRF;
* query/document prefixes не используются;
* модельная revision должна быть закреплена;
* старая collection сохраняется для rollback.

---

# 20. Финальные команды проверки

Выполнить:

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Затем:

```powershell
docker compose ps
```

Проверить:

```text
/health/live
/health/ready
/v1/search
full evaluation
load test
rollback alias
```

После успешной проверки:

```powershell
git status
git add .
git commit -m "feat: promote BGE-M3 to production embeddings"
```

Не добавлять в commit:

```text
скачанные веса модели
Hugging Face cache
локальные benchmark collections
секреты
.env.production с приватными значениями
временные JSON-отчёты, если они игнорируются политикой проекта
```

---

# Definition of Done

* [ ] Production default — `BAAI/bge-m3`.
* [ ] Dimension — 1024.
* [ ] Dtype — fp16 на CUDA.
* [ ] Embeddings нормализуются.
* [ ] Query/document prefixes пустые.
* [ ] Dense и BM25 остаются отдельными named vectors.
* [ ] RRF не изменён.
* [ ] Новая Qdrant collection построена с нуля.
* [ ] Старый индекс не удалён.
* [ ] Runtime metadata совпадает с index metadata.
* [ ] `/health/ready` обнаруживает несовместимый индекс.
* [ ] Exact article работает.
* [ ] Descriptive query работает.
* [ ] Mixed query работает.
* [ ] В выдаче нет дубликатов LD.
* [ ] `LD nDCG@20 >= 0.44`.
* [ ] `LD Recall@20 >= 0.57`.
* [ ] `MRR >= 0.38`.
* [ ] `p95 <= 700 ms`.
* [ ] DN violation rate равен нулю.
* [ ] Все тесты проходят.
* [ ] Ruff проходит.
* [ ] Alias переключён после проверок.
* [ ] Rollback проверен.
* [ ] README обновлён.
* [ ] Рабочее дерево чистое.

---

# Формат отчёта исполнителя

```text
OUTCOME:
Что изменено.

CONFIG:
- model
- revision
- dimension
- device
- dtype
- max sequence length
- batch size
- candidate limit

INDEX:
- collection
- alias
- points
- build time
- CSV SHA256

QUALITY:
- nDCG@20
- Recall@20
- MRR
- nDCG@5
- no-match FP
- DN violations
- duplicate LD rate

PERFORMANCE:
- cold start
- p50
- p95
- p99
- RSS
- VRAM allocated
- VRAM reserved

SMOKE QUERIES:
- запрос
- количество результатов
- top-3 LD
- нарушения

TESTS:
- pytest
- ruff
- format check

ROLLBACK:
- old collection
- new collection
- alias switch verified

COMMIT:
<hash> feat: promote BGE-M3 to production embeddings
```
