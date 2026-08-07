# План доработки `rag_steel`

## 1. Контекст задачи

Репозиторий:

```text
https://github.com/ObiVanBanan/rag_steel
```

В репозитории находится CSV со связями:

```text
товар стороннего производителя → один или несколько аналогов LD
```

Пользователь всегда хочет получить товары LD.

Пользователь может написать:

```text
1184399
а0486
КШ.П.П.015.40-01
Temper DN80 PN16
Кран Broen Ду80 Ру16
Нужен фланцевый кран Temper Ду80 Ру16
Temper 1184399 Ду80
```

Во всех случаях результат должен иметь один формат:

```text
ранжированный список уникальных товаров LD
```

По умолчанию вернуть до 20 уникальных товаров LD.

Если найдено меньше 20 подходящих товаров, вернуть фактическое количество.

---

# 2. Главное архитектурное решение

## 2.1. Не делать маршрутизацию

Запрещено создавать отдельные ветки:

```text
article route
description route
mixed route
```

Запрещено добавлять:

```python
is_article_query
detect_query_type
route_query
search_by_article
search_by_description
```

Не должно быть условий:

```python
if query_is_article:
    ...
else:
    ...
```

В системе должен быть один публичный поисковый метод:

```python
SearchEngine.search(query: str, limit: int = 20)
```

Любой запрос проходит через один pipeline.

## 2.2. Один общий гибридный поиск

Для каждого запроса одновременно выполнять:

1. Нормализацию текста.
2. Создание dense embedding.
3. Создание sparse BM25-представления.
4. Поиск обоими представлениями.
5. Объединение результатов через RRF.
6. Переранжирование найденных кандидатов по характеристикам.
7. Сбор связанных товаров LD.
8. Дедупликацию по артикулу LD.
9. Возврат top-20.

Артикул не является отдельным маршрутом.

Артикул является одним из поисковых сигналов общего запроса.

## 2.3. Что индексировать

Одна точка Qdrant должна соответствовать одному уникальному товару стороннего производителя.

Нельзя индексировать каждую строку CSV как отдельную точку без предварительной группировки.

В payload точки хранить:

* данные товара стороннего производителя;
* список связанных уникальных товаров LD.

Пример:

```json
{
  "steel_id": "stable-hash",
  "steel_name": "Кран шаровой Temper ...",
  "steel_name_variants": [
    "Кран шаровой Temper ...",
    "Шаровой кран Temper ..."
  ],
  "steel_article": "1184399",
  "steel_article_norm": "1184399",
  "steel_brand": "Temper",
  "steel_dn": 80,
  "steel_pn_bar": 16,
  "steel_connection": "фланцевое",
  "steel_medium": "жидкость",
  "steel_control": "ручное",
  "semantic_text": "...",
  "lexical_text": "...",
  "ld_candidates": [
    {
      "ld_article": "11100800162MULD000003000",
      "ld_article_norm": "11100800162muld000003000",
      "ld_name": "Кран шаровый LD ...",
      "ld_url": "...",
      "ld_dn": 80,
      "ld_pn_bar": 16,
      "ld_connection": "фланцевое",
      "ld_medium": "жидкость",
      "ld_control": "ручное",
      "ld_temp": null,
      "ld_length": 300,
      "price_ld": 12130
    }
  ]
}
```

Данные LD не добавлять в поисковый текст исходного товара.

Иначе запрос может найти строку из-за совпадения с характеристиками LD, а не из-за правильного совпадения с товаром конкурента.

---

# 3. Обязательные ограничения для исполнителя

Не добавлять:

* LLM;
* LangChain;
* LangGraph;
* Hermes;
* генерацию ответа;
* query router;
* отдельный поиск артикула;
* отдельный поиск описания;
* отдельный endpoint для аналогов;
* отдельный endpoint сравнения моделей;
* сложную микросервисную архитектуру.

Проект является поисковым сервисом, а не генеративной RAG-системой.

Не изменять исходный CSV.

Не использовать `match_score` в итоговом рейтинге, пока не выяснена его точная семантика.

В текущем CSV встречаются значения:

```text
match_max = 7
match_score = 7, 8 или 9
```

Поэтому нельзя считать:

```python
match_score / match_max
```

корректной вероятностью или нормированным рейтингом.

Не удалять рабочую Qdrant-коллекцию до успешного построения новой.

Не загружать embedding-модель во время импорта Python-модуля.

Не оставлять:

```python
pass
TODO
except Exception:
    pass
```

После каждой фазы запускать тесты и создавать отдельный коммит.

---

# 4. Целевая структура проекта

Не создавать слишком глубокую структуру директорий.

Использовать:

```text
rag_steel/
├── data/
│   ├── mapping_results.csv
│   └── reports/
│       └── data_profile.json
│
├── src/
│   └── rag_steel/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── normalization.py
│       ├── data_builder.py
│       ├── qdrant_index.py
│       ├── query_processor.py
│       ├── ranking.py
│       ├── search_engine.py
│       └── api.py
│
├── tests/
│   ├── test_normalization.py
│   ├── test_data_builder.py
│   ├── test_query_processor.py
│   ├── test_ranking.py
│   ├── test_search_engine.py
│   └── test_api.py
│
├── eval/
│   ├── build_eval_dataset.py
│   ├── evaluate.py
│   └── queries.jsonl
│
├── main.py
├── indexer.py
├── search_engine.py
├── config.py
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── .env.example
├── README.md
└── plan.md
```

Старые файлы в корне оставить как совместимые обёртки.

Например:

```python
# search_engine.py
from rag_steel.search_engine import SearchEngine

__all__ = ["SearchEngine"]
```

---

# 5. Фаза 0. Зафиксировать текущее состояние

## Задачи

1. Создать ветку:

```bash
git checkout -b feature/unified-hybrid-search
```

2. Создать файл:

```text
docs/current_state.md
```

3. Описать существующие проблемы:

* `search_engine.py` пустой;
* `main.py` импортирует несуществующие реализации;
* `SearchEngine` создаётся глобально при импорте;
* текущий индексатор ожидает неправильные имена колонок;
* `build_search_text()` вызывается для исходной строки до правильного маппинга полей;
* путь к CSV захардкожен;
* коллекция удаляется перед созданием новой;
* ошибки удаления коллекции подавляются;
* отсутствует dependency lock;
* отсутствуют тесты;
* `/search` и `/analogs` дублируют смысл;
* `/compare-models` не должен быть production endpoint.

4. Создать один тест, который подтверждает, что текущий проект не может выполнить поиск.

## Критерий приёмки

Команда:

```bash
uv run pytest
```

должна воспроизводимо показывать проблему текущей реализации.

После фиксации проблемы тест можно будет изменить на проверку корректного поведения.

## Коммит

```text
chore: capture current search implementation state
```

---

# 6. Фаза 1. Настроить окружение

## Задачи

Создать `pyproject.toml`.

Использовать Python 3.11.

Основные зависимости:

```text
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pandas
numpy
qdrant-client
sentence-transformers
httpx
```

Dev-зависимости:

```text
pytest
pytest-asyncio
pytest-cov
ruff
mypy
```

Зафиксировать версии в `uv.lock`.

Создать `.env.example`:

```env
APP_ENV=development
LOG_LEVEL=INFO

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_ALIAS=steel_products_active

EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
DENSE_BATCH_SIZE=64
SOURCE_CANDIDATE_LIMIT=300

RESULT_LIMIT_DEFAULT=20
RESULT_LIMIT_MAX=100

WEIGHT_HYBRID=0.45
WEIGHT_TEXT_EXACTNESS=0.20
WEIGHT_SOURCE_FIELDS=0.15
WEIGHT_LD_FIELDS=0.20
```

Создать `docker-compose.yml` только с Qdrant.

Не использовать тег `latest`.

Версию Qdrant и `qdrant-client` выбрать совместимыми и зафиксировать.

## Проверки

```bash
uv sync
docker compose up -d
uv run python -c "import qdrant_client"
uv run python -c "import sentence_transformers"
```

## Коммит

```text
build: add reproducible search environment
```

---

# 7. Фаза 2. Провести профилирование CSV

## Реализовать

```text
src/rag_steel/data_builder.py
```

Добавить функцию:

```python
def profile_csv(csv_path: Path) -> DataProfile:
    ...
```

Проверить наличие колонок:

```text
ld_name
ld_article
ld_url
ld_dn
ld_pn_mpa
ld_connection
ld_medium
ld_control
ld_temp
ld_length
steel_name
steel_article
steel_url
steel_dn
steel_pn_bar
steel_connection
steel_medium
steel_control
steel_temp
steel_length
match_score
match_max
price_ld
```

Отчёт должен содержать:

```text
число строк
число колонок
число полных дублей
число уникальных steel_article
число уникальных ld_article
число уникальных пар steel_article + ld_article
null count для каждой колонки
распределение match_score
распределение match_max
конфликтующие значения для одного артикула
```

Сохранить:

```text
data/reports/data_profile.json
```

Для текущего файла ожидаются приблизительно:

```text
rows: 55539
columns: 23
full_duplicates: 11603
unique_steel_articles: 15708
unique_ld_articles: 3280
unique_steel_ld_pairs: 43719
```

Не использовать эти числа как вечные константы.

Они нужны только как smoke check текущего файла.

## Ошибки

Если отсутствует обязательная колонка:

* вывести понятное сообщение;
* завершить CLI с ненулевым exit code;
* не начинать индексацию.

## Коммит

```text
feat: add source mapping data profiling
```

---

# 8. Фаза 3. Реализовать нормализацию данных

## Файл

```text
src/rag_steel/normalization.py
```

## Функции

```python
normalize_text()
normalize_article()
normalize_brand()
normalize_connection()
normalize_medium()
normalize_control()
normalize_dn()
normalize_pn_bar()
normalize_temperature()
normalize_length()
```

## `normalize_text`

Выполнить:

1. Unicode NFKC.
2. `casefold()`.
3. `ё → е`.
4. Неразрывные пробелы → обычные.
5. Повторяющиеся пробелы → один.
6. Удаление пробелов по краям.

## `normalize_article`

Сохранять:

```text
article_raw
article_norm
article_compact
```

Пример:

```text
КШ.П.П.015.40-01
```

должен дать:

```text
article_raw: КШ.П.П.015.40-01
article_norm: кш.п.п.015.40-01
article_compact: кшпп0154001
```

Правила `article_compact`:

* привести к lower case;
* удалить пробелы;
* удалить точки;
* удалить дефисы;
* удалить `/`, `\`, `_`;
* не удалять буквы;
* не удалять цифры.

Не выполнять полную транслитерацию кириллицы в латиницу.

## PN

В нормализованной внутренней модели использовать:

```text
pn_bar
```

Поле CSV:

```text
ld_pn_mpa
```

по фактическим значениям содержит PN в барах.

Не умножать его повторно на 10.

Для текста запроса поддержать преобразование:

```text
1,6 МПа → 16 бар
2,5 МПа → 25 бар
4 МПа → 40 бар
```

## Бренды

Использовать явный словарь:

```python
{
    "temper": "Temper",
    "broen": "Broen",
    "also": "ALSO",
    "алсо": "ALSO",
    "marshal": "MARSHAL",
    "маршал": "MARSHAL",
    "бивал": "Бивал",
    "bival": "Бивал",
    "forteca": "FORTECA",
}
```

Не использовать эвристику «первое слово с заглавной буквы».

## Тесты

Добавить минимум 30 unit-тестов:

* кириллица;
* латиница;
* смешанный регистр;
* артикулы с точками;
* артикулы с дефисами;
* DN;
* PN;
* МПа;
* пустые значения;
* `NaN`;
* разные варианты брендов.

## Коммит

```text
feat: normalize source and LD product fields
```

---

# 9. Фаза 4. Собрать уникальные товары и связи

## Задача

Преобразовать CSV в список уникальных исходных товаров.

Модель:

```python
class LDProduct(BaseModel):
    article: str
    article_norm: str
    name: str
    url: str | None
    dn: float | None
    pn_bar: float | None
    connection: str | None
    medium: str | None
    control: str | None
    temperature: str | None
    length_mm: float | None
    price: float | None
```

```python
class SteelProductDocument(BaseModel):
    steel_id: str
    article: str
    article_norm: str
    article_compact: str
    name: str
    name_variants: list[str]
    brand: str | None
    dn: float | None
    pn_bar: float | None
    connection: str | None
    medium: str | None
    control: str | None
    temperature: str | None
    length_mm: float | None
    url: str | None
    semantic_text: str
    lexical_text: str
    ld_candidates: list[LDProduct]
```

## Stable ID

Использовать детерминированный hash:

```python
steel_id = sha1(
    article_compact
    + normalized_name
    + normalized_dn
    + normalized_pn
    + normalized_connection
    + normalized_control
).hexdigest()
```

Не использовать индекс DataFrame.

## Группировка

1. Удалить полные дубли строк.
2. Нормализовать поля.
3. Удалить дубли пар:

```text
steel_id + ld_article_norm
```

4. Собрать все LD-аналоги одного `steel_id` в `ld_candidates`.
5. Внутри `ld_candidates` один `ld_article_norm` должен встречаться один раз.
6. Сохранить все варианты `steel_name` в `name_variants`.
7. Отсортировать `ld_candidates` по `ld_article_norm`, чтобы сборка была детерминированной.

## Конфликты

Если у одного артикула отличаются критические характеристики:

```text
DN
PN
connection
control
```

не объединять такие записи только по артикулу.

Они должны получить разные `steel_id`.

## Критерии приёмки

* нет одинаковых `steel_id`;
* внутри одного документа нет повторяющихся LD;
* у каждого документа есть минимум один LD;
* у каждого LD есть `ld_article_norm`;
* повторная сборка даёт те же `steel_id`;
* порядок строк CSV не влияет на результат.

## Коммит

```text
feat: group source products with unique LD candidates
```

---

# 10. Фаза 5. Сформировать поисковые тексты

## Semantic text

Используется для dense embedding.

Включить:

```text
название
бренд
тип товара
DN
PN
соединение
среда
управление
температура
длина
```

Пример:

```text
Кран шаровой Temper.
Диаметр DN 80.
Давление PN 16 бар.
Фланцевое присоединение.
Рабочая среда: жидкость.
Управление: ручное.
```

Не добавлять данные LD.

Артикул можно добавить один раз в конце, но не повторять его многократно.

## Lexical text

Используется для BM25.

Включить:

```text
исходное название
варианты названия
бренд
сырой артикул
нормализованный артикул
компактный артикул
DN80
DN 80
Ду80
Ду 80
PN16
PN 16
Ру16
Ру 16
16 бар
connection
medium
control
```

Пример:

```text
Кран шаровой Temper 1184399 1184399
Temper DN80 DN 80 Ду80 Ду 80
PN16 PN 16 Ру16 Ру 16 16 бар
фланцевое жидкость ручное
```

Повторять технические варианты допустимо, но не создавать десятки одинаковых токенов.

## Проверки

Для строки с артикулом `КШ.П.П.015.40-01` в `lexical_text` должны присутствовать:

```text
КШ.П.П.015.40-01
кш.п.п.015.40-01
кшпп0154001
```

## Коммит

```text
feat: build semantic and lexical product documents
```

---

# 11. Фаза 6. Построить Qdrant-индекс

## Коллекция

Использовать две named vector representations:

```text
dense
sparse
```

`dense`:

* локальная SentenceTransformer-модель;
* cosine distance;
* embedding строится из `semantic_text`.

`sparse`:

* Qdrant BM25;
* строится из `lexical_text`;
* использовать multilingual tokenizer.

## Point

Одна Qdrant point:

```text
один SteelProductDocument
```

Payload должен содержать полный документ, включая `ld_candidates`.

## Индексация

Embeddings создавать батчами:

```python
model.encode(
    texts,
    batch_size=settings.dense_batch_size,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

Запрещено:

```python
for row in rows:
    model.encode(row)
```

## Версионирование

Новая коллекция:

```text
steel_products_<model_slug>_<timestamp>
```

Активный alias:

```text
steel_products_active
```

Порядок:

1. Создать новую коллекцию.
2. Загрузить точки.
3. Проверить количество точек.
4. Проверить случайные payload.
5. Выполнить минимум пять тестовых запросов.
6. Только после этого переключить alias.
7. Не удалять старую коллекцию автоматически.

## Метаданные

Сохранить в отдельной service point или JSON:

```text
CSV SHA256
embedding model
embedding dimension
build timestamp
document count
source row count
deduplicated row count
```

## CLI

Корневой `indexer.py` должен поддерживать:

```bash
uv run python indexer.py \
  --csv data/mapping_results.csv \
  --recreate
```

## Smoke queries

Проверить минимум:

```text
1184399
а0486
Temper DN80 PN16
Broen Ду80 Ру16
фланцевый кран Ду50 Ру40
```

## Коммит

```text
feat: build versioned dense and BM25 index
```

---

# 12. Фаза 7. Реализовать единый QueryProcessor

## Файл

```text
src/rag_steel/query_processor.py
```

## Важно

`QueryProcessor` не определяет тип запроса.

Модель не должна содержать:

```text
route
query_type
is_article_query
```

Использовать:

```python
class ProcessedQuery(BaseModel):
    raw: str
    normalized: str
    compact: str
    semantic_text: str
    lexical_text: str

    possible_article_tokens: list[str]

    brand: str | None
    dn: float | None
    pn_bar: float | None
    connection: str | None
    medium: str | None
    control: str | None
```

Все поля вычисляются для любого запроса.

Пример:

```text
Temper 1184399 Ду80 Ру16
```

Результат:

```json
{
  "raw": "Temper 1184399 Ду80 Ру16",
  "normalized": "temper 1184399 ду80 ру16",
  "compact": "temper1184399ду80ру16",
  "semantic_text": "query: Temper 1184399, кран DN 80 PN 16",
  "lexical_text": "temper 1184399 ду80 dn80 ру16 pn16 16 бар",
  "possible_article_tokens": ["1184399"],
  "brand": "Temper",
  "dn": 80,
  "pn_bar": 16,
  "connection": null,
  "medium": null,
  "control": null
}
```

Это не маршрутизация.

Извлечённые поля используются как дополнительные признаки ранжирования.

## E5

Если в конфигурации выбрана E5-модель:

```text
intfloat/multilingual-e5-base
```

обязательно использовать:

```text
query: ...
passage: ...
```

Для других моделей префиксы применять только при необходимости конкретной модели.

Реализовать это через адаптер:

```python
class EmbeddingTextAdapter:
    def prepare_query(text: str) -> str:
        ...

    def prepare_document(text: str) -> str:
        ...
```

Не разбрасывать проверки имени модели по всему коду.

## Коммит

```text
feat: add unified query normalization
```

---

# 13. Фаза 8. Реализовать один гибридный поиск

## Файл

```text
src/rag_steel/search_engine.py
```

## Публичный интерфейс

```python
class SearchEngine:
    def search(self, query: str, limit: int = 20) -> SearchResponse:
        ...
```

Других публичных поисковых методов не создавать.

Запрещено:

```python
search_article()
search_description()
find_analogs()
```

## Алгоритм

```text
ProcessedQuery
      ↓
dense query embedding
      ↓
BM25 sparse query
      ↓
один Qdrant Query API request
      ↓
RRF fusion
      ↓
до 300 source candidates
```

Псевдокод:

```python
processed = query_processor.process(query)

dense_vector = embedding_model.encode(
    embedding_adapter.prepare_query(processed.semantic_text),
    normalize_embeddings=True,
)

source_hits = qdrant.query_points(
    collection_name=settings.collection_alias,
    prefetch=[
        models.Prefetch(
            query=dense_vector,
            using="dense",
            limit=settings.source_candidate_limit,
        ),
        models.Prefetch(
            query=models.Document(
                text=processed.lexical_text,
                model="qdrant/bm25",
                options={
                    "tokenizer": "multilingual",
                    "language": "none",
                },
            ),
            using="sparse",
            limit=settings.source_candidate_limit,
        ),
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF,
    ),
    limit=settings.source_candidate_limit,
    with_payload=True,
)
```

Адаптировать код под фактическую зафиксированную версию `qdrant-client`.

Не реализовывать два последовательных поиска вручную, если Query API поддерживает fusion.

## Один pipeline для всех запросов

Запрос:

```text
1184399
```

так же создаёт:

* dense query;
* sparse query;
* RRF result.

Запрос:

```text
Temper DN80 PN16
```

делает абсолютно то же самое.

Разница появляется только в значениях признаков, но не в управляющем потоке.

## Коммит

```text
feat: implement one unified hybrid search pipeline
```

---

# 14. Фаза 9. Переранжировать исходные товары

После RRF для каждого найденного исходного товара вычислить дополнительные признаки.

## Признаки

### `hybrid_score`

Оценка Qdrant после fusion.

Привести к диапазону `[0, 1]` внутри текущей выдачи.

Не считать её вероятностью.

### `text_exactness`

Сравнить запрос с:

```text
steel_article
steel_article_norm
steel_article_compact
steel_name
brand
```

Для артикула:

```text
полное compact-совпадение      1.00
полное normalized-совпадение   1.00
prefix                         0.85
contains                       0.75
иначе                          0.00
```

Этот признак вычислять для всех запросов.

Не создавать отдельную ветку поиска.

### `source_field_score`

Сравнить извлечённые поля запроса с исходным товаром:

```text
brand
DN
PN
connection
medium
control
```

Если поле отсутствует в запросе, исключить его из расчёта.

Не ставить за отсутствующее поле ноль.

Использовать среднее только по доступным признакам.

## Начальная формула

```text
source_score =
    weighted_average(
        hybrid_score:       0.55,
        text_exactness:     0.25,
        source_field_score: 0.20
    )
```

Если `text_exactness` неприменим, перенормировать оставшиеся веса.

Например:

```text
hybrid_score 0.55
source_field_score 0.20
```

превращаются в относительные веса:

```text
0.7333
0.2667
```

Не добавлять скрытые магические коэффициенты.

Все веса вынести в config.

## Почему это не роутинг

Система всегда выполняет одинаковые шаги:

```text
hybrid retrieval
text exactness
structured comparison
```

Просто для запроса без артикула `text_exactness` будет отсутствовать или иметь малый вклад.

## Коммит

```text
feat: rerank unified source search results
```

---

# 15. Фаза 10. Собрать и ранжировать товары LD

## Алгоритм

Для каждого source hit:

1. Взять `ld_candidates`.
2. Создать кандидата для каждого товара LD.
3. Сравнить характеристики запроса с характеристиками LD.
4. Вычислить итоговый балл.
5. Объединить одинаковые `ld_article_norm`.
6. Оставить лучший score.
7. Сохранить до трёх лучших подтверждений.
8. Отсортировать.
9. Вернуть top-20.

## `ld_field_score`

Сравнить:

```text
DN
PN
connection
medium
control
```

### DN

```text
точное совпадение → 1.0
несовпадение      → 0.0
```

### PN

Начальное правило:

```text
LD PN == запрошенный PN → 1.0
LD PN > запрошенного PN → 0.85
LD PN < запрошенного PN → 0.0
```

### Connection

```text
совпадение    → 1.0
несовпадение  → 0.0
```

### Medium

```text
совпадение    → 1.0
несовпадение  → 0.0
```

### Control

```text
совпадение    → 1.0
несовпадение  → 0.0
```

Если поле не указано в запросе, исключить его из расчёта.

## Итоговый рейтинг LD

Начальная формула:

```text
final_score =
    0.70 × source_score
  + 0.30 × ld_field_score
```

Если в запросе нет характеристик, по которым можно вычислить `ld_field_score`, использовать:

```text
final_score = source_score
```

API-рейтинг:

```python
relevance_rating = round(final_score * 100, 1)
```

Поле не называть:

```text
probability
confidence
accuracy
```

Это рейтинг релевантности, а не вероятность.

## Дедупликация

Ключ:

```text
ld_article_norm
```

Если один LD найден через несколько товаров конкурентов:

```text
LD score = максимальный final_score
```

Не суммировать оценки.

Плохо:

```python
ld_score = score_1 + score_2 + score_3
```

Хорошо:

```python
ld_score = max(score_1, score_2, score_3)
```

Сохранить evidence:

```json
{
  "source_article": "1184399",
  "source_name": "Кран шаровой Temper ...",
  "source_score": 0.97
}
```

До трёх evidence на один LD.

## Объяснения

Каждый результат должен содержать:

```text
match_reasons
mismatches
source_evidence
score_breakdown
```

Пример:

```json
{
  "relevance_rating": 97.4,
  "match_reasons": [
    "Найден товар Temper с артикулом 1184399",
    "Совпадает DN 80",
    "Совпадает PN 16",
    "Фланцевое присоединение"
  ],
  "mismatches": [],
  "score_breakdown": {
    "hybrid_score": 0.96,
    "text_exactness": 1.0,
    "source_field_score": 1.0,
    "source_score": 0.978,
    "ld_field_score": 0.965,
    "final_score": 0.974
  }
}
```

## Коммит

```text
feat: aggregate and rank unique LD products
```

---

# 16. Фаза 11. Реализовать один API endpoint

## Endpoint

```text
POST /v1/search
```

Request:

```json
{
  "query": "Temper DN80 PN16",
  "limit": 20,
  "include_debug": false
}
```

Response:

```json
{
  "request_id": "uuid",
  "query": "Temper DN80 PN16",
  "count": 20,
  "results": [
    {
      "rank": 1,
      "relevance_rating": 97.4,
      "product": {
        "article": "11100800162MULD000003000",
        "name": "Кран шаровый LD ...",
        "url": "...",
        "price": 12130,
        "dn": 80,
        "pn_bar": 16,
        "connection": "фланцевое",
        "medium": "жидкость",
        "control": "ручное"
      },
      "match_reasons": [
        "Совпадает DN 80",
        "Совпадает PN 16"
      ],
      "mismatches": [],
      "source_evidence": [
        {
          "article": "1184399",
          "name": "Кран шаровой Temper ..."
        }
      ]
    }
  ],
  "timing_ms": {
    "normalize": 0.8,
    "embedding": 12.4,
    "qdrant": 18.2,
    "ranking": 3.5,
    "total": 34.9
  }
}
```

## Удалить дублирующий смысл

Не оставлять одновременно:

```text
/search
/analogs
```

Основной endpoint один:

```text
/v1/search
```

Старые endpoint можно временно оставить как deprecated wrappers, но они обязаны вызывать тот же:

```python
engine.search()
```

Они не должны иметь собственной логики.

## Удалить из API

```text
/compare-models
```

Сравнение моделей выполняется только offline.

## Health

```text
GET /health/live
GET /health/ready
```

`ready` проверяет:

* Qdrant доступен;
* alias существует;
* embedding-модель загружена;
* коллекция содержит точки.

## Lifespan

Использовать FastAPI lifespan.

Не делать:

```python
engine = SearchEngine(...)
```

на уровне импорта.

## Коммит

```text
feat: expose unified LD search API
```

---

# 17. Фаза 12. Создать evaluation dataset

## Цель

Нельзя оценивать качество по двум ручным примерам.

Создать:

```text
eval/queries.jsonl
```

Минимум 500 запросов.

## Категории нужны только для отчёта

Категории eval не являются runtime-маршрутами.

```text
exact_article
modified_article
partial_article
full_name
brand_dn_pn
natural_language
mixed
no_match
```

Runtime не должен знать эти категории.

## Примеры

```json
{
  "query": "1184399",
  "category": "exact_article",
  "expected_ld_articles": [
    "11100800162MULD000003000"
  ]
}
```

```json
{
  "query": "Temper DN80 PN16",
  "category": "brand_dn_pn",
  "expected_ld_articles": [
    "..."
  ]
}
```

## Варианты артикула

Генерировать:

* lower/upper case;
* удаление точек;
* удаление дефисов;
* добавление пробелов;
* compact form;
* уникальный префикс;
* одна опечатка.

Не включать неоднозначный partial article в expected exact result.

## Метрики

```text
LD Recall@5
LD Recall@20
LD Precision@20
LD nDCG@20
MRR
unique LD rate
DN violation rate
PN violation rate
no-match false-positive rate
latency p50
latency p95
```

Обязательная метрика:

```text
unique LD rate = 1.0
```

В ответе не должно быть повторяющихся `ld_article_norm`.

## Целевые пороги первой версии

```text
Exact article LD Recall@20     >= 0.99
Modified article Recall@20     >= 0.95
Unique LD rate                 == 1.00
DN violation rate              == 0
No-match false-positive rate   <= 0.05
```

Для запросов по описанию сначала измерить baseline.

Не придумывать порог качества до получения baseline.

## Коммит

```text
test: add unified LD search evaluation dataset
```

---

# 18. Фаза 13. Сравнить embedding-модели

Сравнить:

```text
paraphrase-multilingual-MiniLM-L12-v2
intfloat/multilingual-e5-base
BAAI/bge-m3
```

Для каждой модели полностью перестроить dense-векторы.

Sparse BM25 оставить одинаковым.

Для каждой модели сохранить:

```text
LD Recall@20
LD nDCG@20
MRR
indexing time
query latency p50
query latency p95
RAM usage
VRAM usage
index size
```

Не выбирать модель по одному запросу.

Не выбирать модель только по публичному leaderboard.

Выбрать модель по следующему порядку:

1. `LD nDCG@20`.
2. `LD Recall@20`.
3. `p95 latency`.
4. Потребление памяти.

Если более тяжёлая модель улучшает качество незначительно, оставить более лёгкую.

## Отчёт

Создать:

```text
eval/model_comparison.md
```

## Коммит

```text
eval: compare multilingual embedding models
```

---

# 19. Фаза 14. Полный набор тестов

## Unit tests

Проверить:

```text
text normalization
article normalization
DN parsing
PN parsing
brand normalization
grouping
stable IDs
semantic text
lexical text
field scoring
LD deduplication
weighted average
score normalization
```

## Integration tests

С тестовым Qdrant проверить:

```text
collection creation
dense upsert
BM25 upsert
hybrid Query API
RRF result
payload loading
alias switching
```

## Search regression tests

Минимум:

```text
1184399
а0486
А0486
а-0486
КШ.П.П.015.40-01
кшпп0154001
Temper DN80 PN16
Temper 1184399 Ду80 Ру16
Broen Ду80 Ру16
фланцевый кран Ду50 Ру40
несуществующий артикул
пустой запрос
```

## API tests

Проверить:

```text
valid request
limit validation
empty query
query too long
Qdrant unavailable
collection missing
zero results
less than 20 results
exactly 20 unique results
no duplicated LD
health live
health ready
```

## Коммит

```text
test: add unified hybrid search regression suite
```

---

# 20. Фаза 15. README и запуск

README должен содержать:

1. Назначение проекта.
2. Что пользователь всегда получает товары LD.
3. Что в runtime нет роутинга запросов.
4. Как работает Dense + BM25 + RRF.
5. Как устроена группировка CSV.
6. Как установить зависимости.
7. Как запустить Qdrant.
8. Как построить индекс.
9. Как запустить API.
10. Примеры запросов.
11. Как запустить тесты.
12. Как запустить evaluation.
13. Как сравнить модели.
14. Как интерпретировать `relevance_rating`.
15. Известные ограничения.

Команды:

```bash
uv sync
docker compose up -d
uv run python indexer.py --csv data/mapping_results.csv --recreate
uv run uvicorn main:app --host 0.0.0.0 --port 8000
uv run pytest
uv run ruff check .
uv run python eval/evaluate.py
```

Пример:

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"1184399","limit":20}'
```

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Temper DN80 PN16","limit":20}'
```

Оба запроса обязаны проходить через один `SearchEngine.search()`.

## Коммит

```text
docs: document unified LD search workflow
```

---

# 21. Definition of Done

Работа закончена только если:

* [ ] Любой запрос проходит через один pipeline.
* [ ] Нет runtime query router.
* [ ] Нет `is_article_query`.
* [ ] Нет отдельных article/description search methods.
* [ ] Каждый запрос одновременно использует Dense и BM25.
* [ ] Результаты объединяются через RRF.
* [ ] Одна Qdrant point соответствует одному исходному товару.
* [ ] В payload находятся связанные товары LD.
* [ ] Полные дубли CSV удаляются до индексации.
* [ ] Дубли связей удаляются.
* [ ] Пользователь всегда получает товары LD.
* [ ] Товары LD дедуплицируются по нормализованному артикулу.
* [ ] Возвращается до 20 уникальных LD.
* [ ] Артикулы с точками и дефисами нормализуются.
* [ ] Запросы по описанию работают.
* [ ] Смешанные запросы работают.
* [ ] `match_score` не используется как ложная вероятность.
* [ ] Есть объяснение рейтинга.
* [ ] Есть evaluation dataset.
* [ ] Есть метрики отдельно по категориям запросов.
* [ ] Категории eval не используются как runtime routes.
* [ ] Embedding-модель выбрана по результатам eval.
* [ ] Qdrant-коллекции версионируются.
* [ ] Alias переключается только после smoke tests.
* [ ] API имеет один основной поисковый endpoint.
* [ ] Все тесты проходят.
* [ ] README-команды проверены.
* [ ] В коде нет заглушек.
* [ ] Рабочее дерево чистое.

---

# 22. Формат работы агента

Выполняй только одну фазу за раз.

После каждой фазы отвечай:

```text
PHASE:
Номер и название фазы.

OUTCOME:
Что реализовано.

ARCHITECTURE CHECK:
- Используется один SearchEngine.search: да/нет
- Добавлена маршрутизация: да/нет
- Результат содержит только LD: да/нет

FILES CHANGED:
- файл
- файл

COMMANDS RUN:
- команда
- команда

TEST RESULTS:
- passed
- failed
- coverage

SEARCH CHECKS:
- запрос
- количество уникальных LD
- первые 3 результата

KNOWN RISKS:
- реальные оставшиеся риски

COMMIT:
<hash> <message>

NEXT PHASE:
Следующая фаза.
```

Если тесты не запускались, не утверждай, что фаза завершена.

Если фаза не проходит тесты:

1. Не переходи к следующей.
2. Исправь текущую фазу.
3. Повторно запусти тесты.
4. Только после успешного результата создай коммит.
