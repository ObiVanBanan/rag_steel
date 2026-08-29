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

Generate one traced request:

```bash
curl -sS -X POST http://127.0.0.1:8005/v2/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"КШ.Ф.П.Р.015.40-01"}'
```

Open Grafana and select the provisioned dashboard:

`RAG Steel — Search Trace`

Paste the response `request_id` into the dashboard `Request ID` filter to get a
per-request stage timeline.

## What the dashboard shows

- traced search count;
- article resolution successes and failures;
- unhandled trace exceptions;
- p95 duration by trace stage;
- article failure codes;
- the full trace timeline filtered by `request_id`;
- article resolution stages, including candidate/dedup information when emitted.

The dashboard intentionally keeps `request_id`, article values, and stage payload
fields as query-time parsed JSON instead of Loki index labels. This avoids
high-cardinality labels.

## Security note

Alloy uses the Docker Engine socket for container discovery and log tailing.
The socket is mounted only into the Alloy container and the config keeps only
the `rag-steel-api` target. Treat access to the Alloy container as privileged
host access.
