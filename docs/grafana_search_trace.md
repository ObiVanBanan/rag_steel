# Grafana search trace

The `/v2/search` structured trace is emitted by the API as JSON log records with
`event="search_trace"`. Grafana reads those logs through this local pipeline:

`rag-steel-api stdout -> Grafana Alloy -> Loki -> Grafana`

## Enable trace emission

Set this in the deployment `.env`:

```env
SEARCH_TRACE_ENABLED=true
```

Then recreate the API container so the environment is reloaded.

## Start observability services

```bash
docker compose up -d loki alloy grafana
docker compose up -d --build --force-recreate api
```

Useful health checks:

```bash
curl -fsS http://127.0.0.1:3100/ready
curl -fsS http://127.0.0.1:12345/-/ready
docker logs --tail 100 rag-steel-alloy
```

Generate one traced batch request:

```bash
curl -sS -X POST http://127.0.0.1:8005/v2/search \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: steel-trace-001' \
  -d '{"products":["КШ.Ф.П.Р.015.40-01","Затвор дисковый DN100 PN16"],"limit":20}'
```

Open Grafana and start with:

- `RAG Steel — Operations` for one-line item decisions, batch outcomes and
  missing required parameters;
- `RAG Steel — Search Trace` for the detailed stage timeline.

Paste the response `request_id` into the dashboard filter.

Paste the response `request_id` into the dashboard `Request ID` filter to get a
per-request stage timeline.

## What the dashboard shows

- traced search count;
- article resolution successes and failures;
- unhandled trace exceptions;
- p95 duration by trace stage;
- article failure codes;
- the full trace timeline filtered by `request_id`;
- article resolution stages, including candidate/dedup information when emitted;
- `required_parameters_ok` / `required_parameters_missing` before retrieval.

The operations dashboard consumes `batch_item_diagnostic` logs and the
`rag_batch_*` / `rag_required_parameter_*` Prometheus metrics. Query text and
request IDs stay in Loki rather than Prometheus labels to avoid high-cardinality
time series.

The dashboard intentionally keeps `request_id`, article values, and stage payload
fields as query-time parsed JSON instead of Loki index labels. This avoids
high-cardinality labels.

## Security note

Alloy uses the Docker Engine socket for container discovery and log tailing.
The socket is mounted only into the Alloy container and the config keeps only
the `rag-steel-api` target. Treat access to the Alloy container as privileged
host access.
