Да. Сейчас я бы **не трогал Grafana/Prometheus**, не трогал load harness и не продолжал архитектурный рефакторинг. Следующий шаг — маленькая изолированная wave:

# Wave: PN minimum-pressure semantics

Ниже можно целиком отдавать более слабой модели.

---

## Контекст

Работаем в `rag_steel`.

Предыдущая большая correctness-wave уже реализовала:

```text
DeepSeek
   ↓
deterministic query resolution
   ↓
brand/article resolution
   ↓
hard validation
   ↓
exact article fast path ИЛИ hard-filtered RAG
   ↓
competitor + LD articles
```

Также уже создан `Eval V4` и датасет на 150 кейсов.

**Эту архитектуру не переделывать.**

Нужно сделать только одно бизнес-изменение:

> PN из запроса означает минимально допустимое рабочее давление товара.

---

# 1. Новая семантика PN

Сейчас PN, скорее всего, в части кода рассматривается как exact constraint:

```text
candidate PN == requested PN
```

Нужно заменить product eligibility на:

```text
candidate PN >= requested PN
```

Пример:

```text
Запрос PN16

candidate PN10  → ❌
candidate PN16  → ✅
candidate PN25  → ✅
candidate PN40  → ✅
candidate PN63  → ✅
```

Другой пример:

```text
Запрос PN25

PN16 → ❌
PN25 → ✅
PN40 → ✅
PN63 → ✅
```

---

# 2. Очень важно: extraction не менять

Если пользователь написал:

```text
Temper DN50 PN16
```

DeepSeek обязан извлечь:

```json
{
  "pn_bar": 16
}
```

Если DeepSeek вернул:

```json
{
  "pn_bar": 25
}
```

это **ошибка extraction**.

Новая семантика касается только проверки товара:

```text
query PN16
candidate PN25

→ технически совместим
```

Не путать:

```text
EXTRACTION
expected == extracted
```

и:

```text
PRODUCT ELIGIBILITY
candidate >= requested
```

---

# 3. Остальные hard constraints не менять

После изменения должно быть:

```text
brand       → exact canonical match
DN          → exact
connection  → exact
PN          → candidate >= requested
```

Если параметр отсутствует в запросе:

```text
DN = null
PN = null
connection = null
```

он остаётся wildcard, как сейчас.

---

# 4. Сначала найти все места сравнения PN

Перед изменениями провести поиск по проекту:

```text
pn_bar
constraints.pn_bar
attributes.pn_bar
requested pn
MatchValue
Range
```

Не заменять вслепую все `==`.

Разделить найденные места на:

```text
A. DeepSeek/eval extraction comparisons
B. product eligibility
C. Qdrant hard filtering
D. article fast-path validation
E. dataset/eval GOLD generation
```

Изменять только B–E.

---

# 5. Python hard validation

Там, где сейчас логика примерно:

```python
if requested_pn is not None and candidate_pn != requested_pn:
    return False
```

заменить на:

```python
if requested_pn is not None:
    if candidate_pn is None:
        return False
    if candidate_pn < requested_pn:
        return False
```

То есть отсутствие PN у товара при явно заданном requested PN — невалидно.

---

# 6. Qdrant hard filter

Это критично.

Недостаточно изменить Python post-validation, если Qdrant заранее отбрасывает PN25 при запросе PN16.

Найти построение filter для:

```text
pn_bar
```

Если сейчас используется exact match вроде:

```python
MatchValue(value=16)
```

для PN использовать numeric range:

```text
gte=requested_pn
```

Логически:

```text
pn_bar >= requested_pn
```

Не менять filters для:

```text
brand
DN
connection
```

---

# 7. Проверить тип поля Qdrant

Перед изменением убедиться, что `pn_bar` в payload реально numeric.

Не пытаться применять numeric range к строке.

Если payload уже содержит:

```json
"pn_bar": 16.0
```

использовать существующее поле.

Не менять schema/index format без необходимости.

---

# 8. Exact article fast path

Это второй критичный участок.

Пример:

```text
article → товар PN25
query   → PN16
```

Это теперь:

```text
25 >= 16
→ compatible
```

Нельзя возвращать `HARD_CONSTRAINT_CONFLICT`.

Но:

```text
article → PN16
query   → PN25
```

это:

```text
16 < 25
→ conflict
→ not_found
```

---

# 9. Article + несколько hard attrs

Пример:

```text
Article X:
brand = Broen
DN = 50
PN = 25
connection = фланцевое
```

Запрос:

```text
X DN50 PN16 фланцевое
```

→ ✅

Запрос:

```text
X DN65 PN16 фланцевое
```

→ ❌ из-за DN.

Запрос:

```text
X DN50 PN40 фланцевое
```

→ ❌ из-за PN.

Запрос:

```text
X DN50 PN16 сварное
```

→ ❌ из-за connection.

То есть PN inequality не должна ослабить остальные hard constraints.

---

# 10. Не вводить PN tolerance

Никакого:

```text
PN25 почти PN40
```

или:

```text
±10%
```

Только:

```text
candidate >= requested
```

---

# 11. Не вводить upper bound

Если пользователь просит:

```text
PN16
```

то:

```text
PN100
```

формально допустим по текущему новому бизнес-правилу.

Не устанавливать самостоятельно:

```text
PN <= 2 * requested
```

или другие эвристики.

---

# 12. Пока НЕ менять ranking

Это важно.

Не добавлять сейчас:

```text
PN16 лучше PN25,
PN25 лучше PN40
```

в ranking.

То есть все:

```text
PN16
PN25
PN40
```

являются eligible.

Существующий RAG сам определяет порядок.

Если после eval окажется, что слишком высокий PN систематически поднимается выше более подходящего — это будет отдельный ranking fix.

---

# 13. Eval V4: DeepSeek matching оставить strict

Найти функцию вроде:

```python
hard_exact_match(...)
```

или аналогичную.

Для extraction:

```text
GOLD PN16
DeepSeek PN16 → ✅

GOLD PN16
DeepSeek PN25 → ❌
```

Эту функцию **не переводить на `>=`**.

---

# 14. Eval V4: product constraint matching изменить

Функция вроде:

```python
matches_hard_constraints(expected, candidate)
```

должна использовать:

```python
if expected.pn_bar is not None:
    if candidate.pn_bar is None:
        return False
    if candidate.pn_bar < expected.pn_bar:
        return False
```

Но:

```text
brand/DN/connection
```

остаются exact.

---

# 15. V4 builder

Текущий V4 dataset был создан **до этой новой бизнес-семантики**.

Поэтому после фикса его необходимо пересобрать.

При построении:

```text
eligible_competitor_articles
```

использовать:

```text
brand exact
DN exact if requested
connection exact if requested
PN >= requested if requested
```

---

# 16. Не менять explicit expected attributes

Если query:

```text
Temper DN50 PN16
```

в GOLD:

```json
{
  "dn": 50,
  "pn_bar": 16
}
```

Не менять expected PN на 25/40.

GOLD attributes описывают **запрос пользователя**, а не найденный товар.

---

# 17. Preferred competitor semantics

На этой wave не вводить правило:

```text
exact PN становится preferred
```

То есть нельзя автоматически сказать:

```text
requested PN16
PN16 = preferred
PN25 = eligible only
```

если раньше soft/preferred logic этого не требовала.

Сначала хотим измерить существующий RAG.

---

# 18. Добавить специальную категорию V4

Если builder позволяет аккуратно расширить dataset, добавить небольшую категорию:

```text
pn_minimum_semantics
```

Примерно 8–12 реальных кейсов.

Не выдумывать products.

Выбирать реальные source products так, чтобы для одного запроса существовали товары:

```text
requested PN
exact PN
higher PN
lower PN
```

где dataset это позволяет.

---

# 19. Обязательные unit tests

Минимум:

```text
requested=16 candidate=10  → false
requested=16 candidate=16  → true
requested=16 candidate=25  → true
requested=16 candidate=40  → true

requested=25 candidate=16  → false
requested=25 candidate=25  → true
requested=25 candidate=40  → true
```

Плюс:

```text
requested=None candidate=10 → true
requested=None candidate=None → true
requested=16 candidate=None → false
```

---

# 20. Qdrant filter test

Нужен отдельный тест, который не просто проверяет итоговый результат, а утверждает, что filter для PN16 означает:

```text
pn_bar >= 16
```

а не:

```text
pn_bar == 16
```

Это защищает от скрытой регрессии.

---

# 21. Article fast-path tests

Минимум:

```text
article PN25 + query PN16 → success

article PN16 + query PN25 → not_found/conflict

article PN25 + no PN → success
```

---

# 22. RAG regression

Обязательно проверить invariant:

```text
candidate PN < requested
```

никогда не попадает в результаты.

Например если запрос PN25, ни один PN16 candidate не должен оказаться даже ниже в TOP-20.

---

# 23. Не менять response schema

Публичный API должен остаться тем же.

Например:

```json
"requested": {
  "pn_bar": 16
}
```

Даже если выбран competitor:

```json
"pn_bar": 25
```

Это нормально.

Не переписывать requested на PN25.

---

# 24. Observability не менять

В этой wave:

```text
НЕ делать Grafana
НЕ делать Prometheus deployment
НЕ делать Loki
```

Текущие `/metrics`, JSON logs и timings оставить.

Если для тестов нужен маленький diagnostic — допустимо, но не делать новую observability architecture.

---

# 25. Load test не запускать

Не запускать:

```text
1,2,5,10,20,50 × 100
```

до завершения correctness baseline.

---

# 26. Полная проверка

После реализации:

```powershell
uv run ruff check src tests eval
uv run pytest -q
```

Не ограничиваться targeted tests.

Сейчас baseline был:

```text
239 passed
```

После добавления новых тестов число должно быть не меньше.

---

# 27. Пересобрать V4 dataset

После зелёных unit tests:

```powershell
uv run python -m eval.build_v4_eval_dataset
```

Показать:

```text
total cases
by category
by brand
pn_minimum_semantics cases
```

И убедиться, что `eligible` изменился там, где появились более высокие PN.

---

# 28. Сначала Resolution/RAG V4

Если они не требуют внешних ключей:

```powershell
uv run python -m eval.evaluate_resolution_v4
uv run python -m eval.evaluate_rag_v4
```

Ожидаемые invariants:

```text
false correction rate = 0
hard violation rate = 0
LD exact = 1.0
```

Не подгонять ranking ради метрик.

---

# 29. DeepSeek/E2E

Если окружение агента снова ловит:

```text
WinError 10013
```

не заниматься обходом сетевой политики.

Просто зафиксировать:

```text
full DeepSeek/E2E not executed because environment cannot reach upstream
```

Пользователь запустит их из рабочего окружения отдельно.

---

# 30. Что вернуть после работы

Отчёт:

```text
PN semantics implemented: yes/no

Production:
- Qdrant PN filter:
- Python eligibility:
- Article validation:

Examples:
requested PN16 / candidate PN10 = ...
requested PN16 / candidate PN16 = ...
requested PN16 / candidate PN25 = ...
requested PN16 / candidate PN40 = ...

Eval:
- extraction remains strict: yes/no
- product eligibility uses >=: yes/no
- V4 GOLD rebuilt: yes/no
- dataset cases:
- pn_minimum cases:

Tests:
ruff =
pytest =

RAG V4:
...

Production ranking changed:
NO

Prometheus/Grafana changed:
NO

Load testing performed:
NO
```

---

# Acceptance criteria

Wave закончена только если одновременно:

```text
✅ PN10 не подходит для requested PN16
✅ PN16 подходит
✅ PN25 подходит
✅ PN40 подходит

✅ DeepSeek всё ещё должен извлекать именно PN16
✅ article PN25 совместим с requested PN16
✅ article PN16 несовместим с requested PN25
✅ Qdrant использует PN gte, а не equality
✅ DN остаётся exact
✅ connection остаётся exact
✅ brand остаётся exact
✅ hard violation rate остаётся 0
✅ ranking не тюнили
✅ Grafana/Prometheus не трогали
```

---

# И наша общая дорога к production

Да, у нас уже вырисовалась вполне конкретная последовательность. Я бы сейчас зафиксировал её вот так.

## Этап 1 — Correctness ядра

Почти закончили:

```text
✅ normalized product documents
✅ OpenAI embeddings
✅ Qdrant
✅ dense + BM25 + RRF
✅ hard constraints
✅ competitor → LD grouping
✅ body material
✅ DeepSeek extraction

✅ DeepSeek-first Query Resolution
✅ typo-safe brand resolution
✅ article-only search
✅ article typo resolution
✅ identity conflict protection

➡️ PN minimum-pressure semantics     ← МЫ ЗДЕСЬ
➡️ Full Eval V4
```

Цель:

> Сначала доказать, что сервис выдаёт правильные ответы.

---

# Этап 2 — Quality baseline

После PN fix:

```text
DeepSeek-only V4
Resolution-only V4
RAG-only V4
E2E V4
```

Причём разделение очень полезное:

```text
USER QUERY
   ↓
DeepSeek        ← измеряем отдельно
   ↓
Resolver        ← измеряем отдельно
   ↓
RAG             ← измеряем отдельно
   ↓
E2E             ← измеряем всё вместе
```

Так мы всегда понимаем, **где именно упало качество**.

После full V4 фиксируем baseline и дальше без причины correctness pipeline не трогаем.

---

# Этап 3 — Load / Stress

Потом берём уже написанный harness:

```text
C=1
  ↓
C=2
  ↓
C=5
  ↓
C=10
  ↓
C=20
  ↓
C=50
```

Измеряем:

```text
RPS
successful RPS

p50
p90
p95
p99

DeepSeek
Embedding
Qdrant
Ranking

SERVICE_BUSY
503
timeouts

in-flight
readiness
```

И ищем реальную saturation point.

Не предполагаем заранее, что сервис выдержит 20 или 50.

---

# Этап 4 — Runtime protection tuning

Runtime protection у нас уже есть как baseline.

После load test смотрим факты.

Например окажется:

```text
C1   → 0.6 RPS
C2   → 1.2 RPS
C5   → 2.9 RPS
C10  → 4 RPS
C20  → 4.1 RPS + latency x3
```

Тогда разумно ограничить:

```text
max_concurrent_searches ≈ 8–10
```

а не гадать.

Настраиваем:

```text
concurrency gate
timeouts
retries
Retry-After
graceful overload
```

Но не архитектуру поиска.

---

# Этап 5 — Deployment hardening

После этого превращаем приложение в нормальный deployable service.

Примерно:

```text
           Client
              │
              ▼
      reverse proxy / gateway
              │
              ▼
       RAG Steel API
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
   DeepSeek  OpenAI   Qdrant
             Embed
```

Прокси к OpenAI оставляем, как уже решили.

Здесь уже:

```text
Docker image
environment config
secrets
restart policy
health checks
resource limits
graceful shutdown
startup/readiness
```

---

# Этап 6 — Qdrant production safety

У нас уже есть:

```text
versioned physical collection
        ↓
alias
        ↓
active collection
```

и hot alias switching.

Дальше нужны:

```text
backup
restore test
rollback procedure
index rebuild procedure
```

То есть если новый индекс плохой:

```text
collection_v2 ❌
        ↓
alias → collection_v1
```

без пересборки production на месте.

---

# Этап 7 — CI/CD

Потом:

```text
push/PR
 ↓
ruff
 ↓
pytest
 ↓
build image
 ↓
possibly Eval smoke
 ↓
deploy
 ↓
readiness
```

Full дорогостоящий DeepSeek E2E не обязательно гонять на каждый коммит.

Можно разделить:

```text
PR:
unit + integration + small eval

release:
larger correctness suite
```

---

# Этап 8 — API contract freeze

Когда pipeline стабилен:

```text
POST /v2/search
```

фиксируем.

Документируем:

```text
ok
not_found
cannot_process
retryable technical failures

requested
competitor
ld_articles
reason codes
```

После freeze другие сервисы уже спокойно интегрируются с нами.

---

# Этап 9 — Observability

Текущая база уже есть:

```text
structured JSON logs
request_id
/metrics
/health/live
/health/ready
stage timings
```

**Grafana/Prometheus пока откладываем**, как ты попросил.

Когда core и load готовы — вернёмся.

---

# Этап 10 — Runbook

Перед тем как сказать «готово в прод»:

нужен короткий документ:

```text
Как запустить
Как остановить
Как проверить health
Как понять, что умер DeepSeek
Как понять, что умер OpenAI
Как понять, что умер Qdrant
Как сменить индекс
Как откатить индекс
Как восстановить Qdrant
Что делать при SERVICE_BUSY
Где смотреть request_id
```

Это скучная часть, но именно она отличает «работает на моей машине» от сервиса.

---

# И только потом вопрос Kafka

Вот Kafka у нас **не плановый обязательный компонент**.

Решение принимаем после load test.

Если получается:

```text
обычный synchronous HTTP
+
достаточный RPS
+
разумный p95
+
graceful SERVICE_BUSY
```

то:

```text
Kafka не нужна.
```

Если же продукт требует:

```text
1000 запросов прилетели
↓
мы обязаны все принять
↓
обрабатывать несколько минут
↓
клиент может потом получить результат
```

тогда архитектура меняется:

```text
Client
  ↓
API
  ↓
Queue
  ↓
Workers
  ↓
DeepSeek/OpenAI/Qdrant
  ↓
Result store
```

И вот **тогда** Kafka/RabbitMQ/Redis queue становится предметным решением.

---

## Где мы сейчас

Если ужать всю нашу дорогу до одной линии:

```text
SEARCH QUALITY
    ✅
     ↓
QUERY RESOLUTION
    ✅
     ↓
PN >= requested
    ← СЕЙЧАС
     ↓
FULL V4 EVAL
     ↓
LOAD / STRESS
     ↓
RUNTIME TUNING
     ↓
DEPLOYMENT HARDENING
     ↓
QDRANT BACKUP / ROLLBACK
     ↓
CI/CD
     ↓
API FREEZE
     ↓
OBSERVABILITY UI (если хотим)
     ↓
RUNBOOK
     ↓
PRODUCTION
```

И я бы сейчас очень старался **не добавлять новые архитектурные идеи между этими этапами**, пока мы не закончили PN + V4. Мы наконец дошли до состояния, где основная архитектура поиска уже практически сложилась, и дальше нам больше нужно её измерять и укреплять, чем продолжать перестраивать.
