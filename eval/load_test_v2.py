"""Run a concurrency load test against the V2 search API."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import httpx

import main

DEFAULT_QUERIES_PATH = Path("eval/v2_queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORT_PATH = Path("eval/load_test_v2_report.md")
DEFAULT_CONCURRENCY_LEVELS = [1, 5, 10, 20, 50]
DEFAULT_REQUESTS_PER_LEVEL = 20
DEFAULT_LIMIT = 5


@dataclass(slots=True)
class LoadTestObservation:
    query: str
    status_code: int
    latency_ms: float
    response_status: str | None = None
    error_code: str | None = None
    timing_ms: dict[str, float] | None = None
    error_message: str | None = None


@dataclass(slots=True)
class LoadTestLevelResult:
    concurrency: int
    request_count: int
    success_count: int
    busy_count: int
    timeout_count: int
    error_count: int
    rps: float
    wall_ms: float
    client_p50_ms: float
    client_p95_ms: float
    client_p99_ms: float
    server_total_p50_ms: float | None
    server_total_p95_ms: float | None
    server_embedding_p50_ms: float | None
    server_qdrant_p50_ms: float | None
    server_ranking_p50_ms: float | None
    server_total_mean_ms: float | None
    status_counts: dict[str, int]
    error_examples: list[str]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    rank = (len(values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _load_queries(path: Path) -> list[str]:
    queries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        query = str(payload["query"]).strip()
        if query:
            queries.append(query)
    if not queries:
        raise RuntimeError(f"No queries found in {path}")
    return queries


async def _perform_request(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
) -> LoadTestObservation:
    started = perf_counter()
    try:
        response = await client.post("/v2/search", json={"query": query, "limit": limit})
        latency_ms = (perf_counter() - started) * 1000.0
    except httpx.TimeoutException as exc:
        return LoadTestObservation(
            query=query,
            status_code=504,
            latency_ms=(perf_counter() - started) * 1000.0,
            error_code="TIMEOUT",
            error_message=str(exc),
        )

    response_status: str | None = None
    timing_ms: dict[str, float] | None = None
    error_code: str | None = None
    error_message: str | None = None

    try:
        payload = response.json()
    except Exception:
        payload = {}

    if response.status_code == 200 and isinstance(payload, dict):
        response_status = payload.get("status")
        raw_timing = payload.get("timing_ms")
        if isinstance(raw_timing, dict):
            timing_ms = {
                key: float(value)
                for key, value in raw_timing.items()
                if isinstance(value, (int, float))
            }
    elif isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error_code = str(error.get("code") or "")
            error_message = str(error.get("message") or "")

    return LoadTestObservation(
        query=query,
        status_code=response.status_code,
        latency_ms=latency_ms,
        response_status=response_status,
        error_code=error_code,
        timing_ms=timing_ms,
        error_message=error_message,
    )


def _summarize_level(
    concurrency: int,
    observations: list[LoadTestObservation],
    wall_ms: float,
) -> LoadTestLevelResult:
    latencies = [observation.latency_ms for observation in observations]
    success = [observation for observation in observations if observation.status_code == 200]
    busy = [observation for observation in observations if observation.error_code == "SERVICE_BUSY"]
    timeout = [observation for observation in observations if observation.error_code == "TIMEOUT"]
    errors = [
        observation
        for observation in observations
        if observation.status_code != 200
        and observation.error_code not in {"SERVICE_BUSY", "TIMEOUT"}
    ]

    status_counts: dict[str, int] = {}
    for observation in observations:
        key = observation.response_status or f"HTTP_{observation.status_code}"
        status_counts[key] = status_counts.get(key, 0) + 1

    server_total = [
        obs.timing_ms["total"]
        for obs in success
        if obs.timing_ms and "total" in obs.timing_ms
    ]
    server_embedding = [
        obs.timing_ms["embedding"]
        for obs in success
        if obs.timing_ms and "embedding" in obs.timing_ms
    ]
    server_qdrant = [
        obs.timing_ms["qdrant"]
        for obs in success
        if obs.timing_ms and "qdrant" in obs.timing_ms
    ]
    server_ranking = [
        obs.timing_ms["ranking"]
        for obs in success
        if obs.timing_ms and "ranking" in obs.timing_ms
    ]

    error_examples = []
    for observation in observations:
        if observation.status_code == 200:
            continue
        label = observation.error_code or f"HTTP_{observation.status_code}"
        detail = observation.error_message or observation.response_status or observation.query
        error_examples.append(f"{label}: {detail}")
        if len(error_examples) >= 3:
            break

    return LoadTestLevelResult(
        concurrency=concurrency,
        request_count=len(observations),
        success_count=len(success),
        busy_count=len(busy),
        timeout_count=len(timeout),
        error_count=len(errors),
        rps=(len(observations) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
        wall_ms=wall_ms,
        client_p50_ms=_percentile(latencies, 50) or 0.0,
        client_p95_ms=_percentile(latencies, 95) or 0.0,
        client_p99_ms=_percentile(latencies, 99) or 0.0,
        server_total_p50_ms=_percentile(server_total, 50),
        server_total_p95_ms=_percentile(server_total, 95),
        server_embedding_p50_ms=_percentile(server_embedding, 50),
        server_qdrant_p50_ms=_percentile(server_qdrant, 50),
        server_ranking_p50_ms=_percentile(server_ranking, 50),
        server_total_mean_ms=mean(server_total) if server_total else None,
        status_counts=status_counts,
        error_examples=error_examples,
    )


async def _run_level(
    client: httpx.AsyncClient,
    queries: list[str],
    *,
    concurrency: int,
    request_count: int,
    limit: int,
) -> LoadTestLevelResult:
    semaphore = asyncio.Semaphore(concurrency)
    query_cycle = cycle(queries)
    observations: list[LoadTestObservation] = []

    async def worker() -> None:
        query = next(query_cycle)
        async with semaphore:
            observation = await _perform_request(client, query, limit=limit)
            observations.append(observation)

    started = perf_counter()
    await asyncio.gather(*(worker() for _ in range(request_count)))
    wall_ms = (perf_counter() - started) * 1000.0
    return _summarize_level(concurrency, observations, wall_ms)


def _render_report(
    *,
    dataset_path: Path,
    base_url: str,
    limit: int,
    levels: list[LoadTestLevelResult],
) -> str:
    total_requests = sum(level.request_count for level in levels)
    total_success = sum(level.success_count for level in levels)
    total_busy = sum(level.busy_count for level in levels)
    total_timeout = sum(level.timeout_count for level in levels)
    total_errors = sum(level.error_count for level in levels)
    client_latencies = [level.client_p95_ms for level in levels if level.client_p95_ms > 0]

    lines = [
        "# V2 Load Test",
        "",
        "## Run",
        f"- base url: `{base_url}`",
        f"- dataset: `{dataset_path}`",
        f"- limit: `{limit}`",
        f"- total requests: `{total_requests}`",
        "",
        "## Totals",
        f"- success: `{total_success}`",
        f"- busy (503): `{total_busy}`",
        f"- timeout: `{total_timeout}`",
        f"- other errors: `{total_errors}`",
        f"- average p95 client latency across levels: `{mean(client_latencies):.1f}` ms"
        if client_latencies
        else "- average p95 client latency across levels: `0.0` ms",
        "",
        "## Levels",
        "",
        (
            "| Concurrency | Requests | RPS | Client p50 | Client p95 | Client p99 | "
            "Server p50 | Server p95 | Busy | Timeout | Errors |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for level in levels:
        lines.append(
            "| "
            f"{level.concurrency} | {level.request_count} | {level.rps:.2f} | "
            f"{level.client_p50_ms:.1f} | {level.client_p95_ms:.1f} | {level.client_p99_ms:.1f} | "
            f"{(level.server_total_p50_ms or 0.0):.1f} | "
            f"{(level.server_total_p95_ms or 0.0):.1f} | "
            f"{level.busy_count} | {level.timeout_count} | {level.error_count} |"
        )
        if level.error_examples:
            lines.append(
                f"- concurrency `{level.concurrency}` examples: "
                + "; ".join(level.error_examples)
            )

    lines.append("")
    lines.append("## Status Mix")
    for level in levels:
        lines.append(
            f"- `{level.concurrency}`: "
            f"{json.dumps(level.status_counts, ensure_ascii=False)}"
        )

    return "\n".join(lines).rstrip() + "\n"


async def _run_load_test(
    *,
    queries: list[str],
    levels: list[int],
    requests_per_level: int,
    limit: int,
    base_url: str | None,
) -> dict[str, Any]:
    if base_url is None:
        transport = httpx.ASGITransport(app=main.app)
        client_kwargs: dict[str, Any] = {"transport": transport, "base_url": "http://testserver"}
        context = main.app.router.lifespan_context(main.app)
    else:
        client_kwargs = {"base_url": base_url}
        context = None

    results: list[LoadTestLevelResult] = []
    async with httpx.AsyncClient(timeout=120.0, **client_kwargs) as client:
        if context is None:
            for concurrency in levels:
                request_count = max(requests_per_level, concurrency)
                result = await _run_level(
                    client,
                    queries,
                    concurrency=concurrency,
                    request_count=request_count,
                    limit=limit,
                )
                results.append(result)
        else:
            async with context:
                for concurrency in levels:
                    request_count = max(requests_per_level, concurrency)
                    result = await _run_level(
                        client,
                        queries,
                        concurrency=concurrency,
                        request_count=request_count,
                        limit=limit,
                    )
                    results.append(result)

    total_requests = sum(result.request_count for result in results)
    total_success = sum(result.success_count for result in results)
    total_busy = sum(result.busy_count for result in results)
    total_timeout = sum(result.timeout_count for result in results)
    total_errors = sum(result.error_count for result in results)
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "base_url": base_url or "asgi://main.app",
        "dataset_path": "eval/v2_queries.jsonl",
        "limit": limit,
        "levels": [asdict(result) for result in results],
        "summary": {
            "total_requests": total_requests,
            "success_count": total_success,
            "busy_count": total_busy,
            "timeout_count": total_timeout,
            "error_count": total_errors,
            "error_rate": (
                (total_busy + total_timeout + total_errors) / total_requests
                if total_requests
                else 0.0
            ),
        },
    }
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a V2 load test.")
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY_LEVELS,
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        help="Minimum number of requests per level",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional HTTP base URL. Defaults to in-process ASGI transport.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the markdown report",
    )
    return parser


def main_cli(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    queries = _load_queries(args.queries_path)
    report = asyncio.run(
        _run_load_test(
            queries=queries,
            levels=list(args.levels),
            requests_per_level=args.requests_per_level,
            limit=args.limit,
            base_url=args.base_url,
        )
    )

    output_json = args.output_json
    if output_json is None:
        output_json = DEFAULT_RESULTS_DIR / f"load_v2_{report['run_id']}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = _render_report(
        dataset_path=args.queries_path,
        base_url=report["base_url"],
        limit=args.limit,
        levels=[LoadTestLevelResult(**item) for item in report["levels"]],
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(str(output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
