Да. Ниже план, который можно **целиком отдавать более слабой модели**. Цель — **закрыть PN semantics и довести Eval V4 до полного baseline**, без Grafana, без load test и без новых архитектурных идей.

# План: закончить PN semantics + Full Eval V4

## Контекст

Текущая архитектура уже реализована:

```text
USER QUERY
   ↓
DeepSeek extraction
   ↓
Deterministic resolution
   ├─ brand exact/fuzzy
   └─ article exact/fuzzy
   ↓
identity/hard validation
   ↓
exact article fast path
ИЛИ
hard-filtered RAG
   ↓
Dense + BM25 + RRF
   ↓
competitor + ld_articles[]
```

Также уже:

```text
✅ V4 eval infrastructure
✅ V4 dataset builder
✅ текущий V4 dataset ~150 cases
✅ query resolver tests
✅ extractor tests
✅ search tests
✅ full pytest был зелёным
```

Не переделывать это.

---

# Часть A. PN minimum-pressure semantics

## A1. Бизнес-правило

Изменить только семантику product eligibility для PN:

```text
candidate.pn_bar >= requested.pn_bar
```

Примеры:

```text
requested PN16

candidate PN10 → ❌
candidate PN16 → ✅
candidate PN25 → ✅
candidate PN40 → ✅
```

```text
requested PN25

candidate PN16 → ❌
candidate PN25 → ✅
candidate PN40 → ✅
```

---

## A2. DeepSeek extraction НЕ менять

Если пользователь написал:

```text
Temper DN50 PN16
```

DeepSeek должен вернуть:

```json
{
  "pn_bar": 16
}
```

Не:

```json
{
  "pn_bar": 25
}
```

То есть:

```text
EXTRACTION:
expected PN == extracted PN
```

остаётся strict.

Новая логика относится только к товару:

```text
PRODUCT ELIGIBILITY:
candidate PN >= requested PN
```

---

## A3. Остальные hard constraints оставить strict

Должно быть:

```text
brand       → exact resolved brand
DN          → exact
connection  → exact
PN          → candidate >= requested
```

Если поле отсутствует в запросе:

```text
None = wildcard
```

---

## A4. Найти все PN comparisons

Перед изменением найти по проекту:

```text
pn_bar
constraints.pn_bar
attributes.pn_bar
MatchValue
Range
matches_hard_constraints
```

Разделить найденное:

```text
1. extraction correctness
2. product hard validation
3. Qdrant filter
4. article fast path
5. V4 GOLD builder
6. V4 metrics
```

Не делать глобальную замену `==` на `>=`.

---

## A5. Python hard validation

Логика должна быть:

```python
if requested_pn is not None:
    if candidate_pn is None:
        return False
    if candidate_pn < requested_pn:
        return False
```

---

## A6. Qdrant hard filter

Это обязательно.

Если сейчас PN строится как exact:

```python
MatchValue(value=requested_pn)
```

заменить только для PN на numeric range:

```python
Range(gte=requested_pn)
```

Итог:

```text
brand      → MatchValue
dn         → MatchValue
connection → MatchValue
pn_bar     → Range(gte=...)
```

Не менять Qdrant index schema, если `pn_bar` уже numeric.

---

## A7. Exact article fast path

Если resolved article имеет:

```text
PN25
```

а пользователь запросил:

```text
PN16
```

результат совместим:

```text
25 >= 16 → ✅
```

Но:

```text
article PN16
requested PN25

16 < 25 → ❌
```

Ожидание:

```text
not_found / HARD_CONSTRAINT_CONFLICT
```

в соответствии с уже существующим API behavior.

---

## A8. Не вводить PN ranking

На этой wave нельзя добавлять:

```text
PN16 > PN25 > PN40
```

в RRF/ranking.

Все `PN >= requested` являются eligible.

Если позже Eval покажет, что PN63 постоянно вытесняет PN16, это будет отдельная ranking-wave.

---

# Часть B. Тесты PN

## B1. Unit product eligibility

Добавить минимум:

```text
requested=16 candidate=10   → false
requested=16 candidate=16   → true
requested=16 candidate=25   → true
requested=16 candidate=40   → true

requested=25 candidate=16   → false
requested=25 candidate=25   → true
requested=25 candidate=40   → true

requested=None candidate=16 → true
requested=16 candidate=None → false
```

---

## B2. Qdrant filter test

Проверить именно структуру filter.

Для:

```text
requested PN16
```

должно получаться:

```text
pn_bar >= 16
```

а не:

```text
pn_bar == 16
```

---

## B3. Article tests

Добавить:

```text
article PN25 + query PN16 → success
article PN16 + query PN25 → conflict/not_found
article PN25 + query no PN → success
```

И обязательно комбинации:

```text
article DN50 PN25 + query DN50 PN16 → ✅
article DN50 PN25 + query DN65 PN16 → ❌
article DN50 PN25 + query DN50 PN40 → ❌
```

---

# Часть C. Довести V4 dataset

## C1. Не уничтожать V3

Не менять:

```text
eval/v3_*
```

V3 остаётся историческим baseline.

---

## C2. Пересобрать V4 после PN fix

Текущий V4 GOLD был создан до новой PN semantics.

После production fix пересобрать:

```powershell
uv run python -m eval.build_v4_eval_dataset
```

---

## C3. V4 product eligibility

В builder функция определения eligible должна быть:

```text
brand exact
AND DN exact if specified
AND connection exact if specified
AND candidate PN >= requested PN if specified
```

---

## C4. Expected attributes не менять

Для запроса:

```text
Temper DN50 PN16
```

GOLD должен всё ещё содержать:

```json
{
  "brand": "Temper",
  "dn": 50,
  "pn_bar": 16
}
```

Не переписывать PN на PN25 или PN40 только потому, что они допустимы.

---

## C5. `hard_exact_match()` оставить strict

DeepSeek eval:

```text
expected PN16
actual PN16 → ✅
actual PN25 → ❌
```

---

## C6. `matches_hard_constraints()` изменить

Product/RAG eval:

```text
requested PN16

candidate PN10 → ❌
candidate PN16 → ✅
candidate PN25 → ✅
candidate PN40 → ✅
```

Это две разные функции и две разные семантики.

---

# Часть D. Добавить PN-specific eval cases

Добавить категорию:

```text
pn_minimum_semantics
```

Примерно:

```text
8–12 cases
```

Только на реальных товарах из source dataset.

Желательно подобрать случаи, где существует:

```text
requested PN
↓
exact PN candidate
higher PN candidate
lower PN candidate
```

Например логически:

```text
query DN50 PN16
eligible:
  DN50 PN16
  DN50 PN25
  DN50 PN40

not eligible:
  DN50 PN10
```

Не выдумывать артикулы.

---

# Часть E. Проверить состав V4

После builder вывести статистику.

Обязательно показать:

```text
total cases
by category
by brand

article_only_exact
article_only_normalized
article_only_typo
brand_typo
brand_plus_article
article_plus_hard
article_natural_language
unknown_article
ambiguous_article_typo
brand_article_conflict
article_hard_conflict
pn_minimum_semantics

ADL cases
negative cases
```

Также проверить, что нет сильного случайного перекоса в один бренд.

---

# Часть F. Полная локальная validation

После всех изменений:

```powershell
uv run ruff check src tests eval
```

Затем:

```powershell
uv run pytest -q
```

Предыдущий baseline:

```text
239 passed
```

Новый должен быть как минимум не меньше с учётом добавленных тестов.

Если старый тест падает из-за того, что он явно ожидал:

```text
PN == requested PN
```

его можно изменить под новое бизнес-правило.

Не менять unrelated tests.

---

# Часть G. Full Resolution V4

Запустить:

```powershell
uv run python -m eval.evaluate_resolution_v4
```

Особенно проверить:

```text
brand exact accuracy
brand fuzzy accuracy
article exact accuracy
article fuzzy accuracy
ambiguity accuracy
identity conflict accuracy
false correction rate
```

Критический invariant:

```text
false correction rate = 0
```

Лучше unresolved, чем неверный товар.

---

# Часть H. Full RAG V4

Запустить:

```powershell
uv run python -m eval.evaluate_rag_v4
```

Критические метрики:

```text
status accuracy
hard violation rate
eligible hit@1
eligible hit@5
preferred hit@1
preferred hit@5
MRR
LD exact
invalid competitor rate
```

Главное:

```text
hard violation rate = 0
```

Ни один:

```text
candidate PN < requested PN
```

не должен попадать в выдачу.

---

# Часть I. DeepSeek V4

Если есть рабочий upstream:

```powershell
uv run python -m eval.evaluate_deepseek_v4
```

Смотреть:

```text
raw_brand accuracy
article accuracy
DN accuracy
PN accuracy
connection accuracy
hallucination rate
missing rate
latency
```

PN extraction всё ещё оценивается strict.

---

# Часть J. E2E V4

После успешных предыдущих слоёв:

```powershell
uv run python -m eval.evaluate_e2e_v4
```

Проверить весь путь:

```text
query
↓
DeepSeek
↓
resolver
↓
PN/DN/brand/connection hard validation
↓
article fast path / RAG
↓
competitor
↓
LD
```

Метрики:

```text
status accuracy
eligible hit@1/@5
preferred hit@1/@5
invalid competitor rate
overall pass
strict overall pass
```

---

# Часть K. Compare

После всех full runs:

```powershell
uv run python -m eval.compare_v4_results
```

Сформировать:

```text
eval/v4_summary.md
```

---

# Если DeepSeek/E2E снова ловят WinError 10013

Не чинить сеть.

Не менять proxy.

Не менять endpoint.

Не менять DeepSeek client.

Просто написать:

```text
Resolution V4: completed
RAG V4: completed

DeepSeek V4: not executed due to environment network restriction
E2E V4: not executed due to environment network restriction
```

Пользователь прогонит эти два eval из рабочего окружения.

---

# Что категорически НЕ делать

В этой wave:

```text
❌ Grafana
❌ Prometheus deployment
❌ Loki
❌ Kafka
❌ load test
❌ runtime tuning
❌ embedding changes
❌ RRF changes
❌ reranker
❌ candidate-limit tuning
❌ DeepSeek prompt optimization по результатам одного failure
❌ Qdrant rearchitecture
```

Сейчас только:

```text
PN semantics
+
V4 correctness baseline
```

---

# Acceptance criteria

Wave закончена только если:

```text
✅ requested PN16 допускает PN16/25/40
✅ requested PN16 не допускает PN10

✅ Qdrant PN filter использует gte
✅ Python eligibility использует >=
✅ article validation использует >=

✅ DeepSeek extraction остаётся strict
✅ DN exact
✅ brand exact after resolution
✅ connection exact

✅ V4 GOLD пересобран
✅ pn_minimum_semantics покрыт тестами
✅ full pytest green
✅ ruff green

✅ Resolution V4 completed
✅ RAG V4 completed

✅ hard violation = 0
✅ false correction = 0

✅ DeepSeek/E2E full выполнены,
   если рабочее окружение имеет network access
```

---

# Что вернуть мне после выполнения

Пусть модель **не просто пишет “готово”**, а принесёт конкретно:

```text
Commit / working tree state:
...

Changed files:
...

PN implementation:
Qdrant:
Python validation:
Article path:

Examples:
PN16 → PN10 = ...
PN16 → PN16 = ...
PN16 → PN25 = ...
PN16 → PN40 = ...

V4 dataset:
total =
by category =
by brand =
ADL =
pn_minimum_semantics =
negative cases =

Tests:
ruff =
pytest =

Resolution V4:
overall =
brand fuzzy =
article fuzzy =
ambiguity =
conflict =
false correction =

RAG V4:
status =
hard violation =
eligible hit@1 =
eligible hit@5 =
preferred hit@1 =
preferred hit@5 =
LD exact =

DeepSeek V4:
...

E2E V4:
...

Failures:
<каждый failure отдельно>

Ranking changed: NO
Grafana/Prometheus changed: NO
Load test performed: NO
```

После этого уже **мы с тобой разбираем full V4 failures и фиксируем correctness baseline**. И только потом идём в load/stress wave.
