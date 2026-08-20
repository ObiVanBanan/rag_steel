"""Run a production-style concurrency load test against the V5 search API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import httpx

import main
from eval.v5_constants import DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_RESULTS_DIR
from eval.v5_schema import EvalCase
from rag_steel.observability import REQUEST_ID_HEADER
from rag_steel.settings import get_settings

DEFAULT_REPORT_PATH = Path("eval/load_v5_report.md")
DEFAULT_RESULTS_JSON_PATH = DEFAULT_RESULTS_DIR / "load_v5_latest.json"
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 6, 8]
DEFAULT_REQUESTS_PER_STAGE = 100
DEFAULT_WARMUP_REQUESTS = 5
DEFAULT_LIMIT = 5
DEFAULT_WORKLOADS = ["full_pipeline"]
DEFAULT_CLIENT_TIMEOUT_SECONDS = 180.0
DEFAULT_ENDPOINT = "/v2/search"
MAX_CONCURRENCY_LIMIT = 100
MAX_REQUESTS_PER_STAGE = 1000
PREFLIGHT_REQUEST_COUNT = 3
ALLOWED_RESULT_STATUSES = {"exact_match", "not_found", "cannot_process"}
WORKLOADS = ("full_pipeline", "article_fast_path", "business_fast_fail", "mixed")


@dataclass(slots=True)
class LoadTestObservation:
    request_number: int
    concurrency: int
    workload_class: str
    query_id: str
    category: str
    expected_status: str
    query: str
    status_code: int
    http_outcome: str
    result_status: str | None = None
    latency_ms: float = 0.0
    server_timing_ms: dict[str, float] | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    malformed_json: bool = False
    invalid_response: bool = False
    status_mismatch: bool = False


@dataclass(slots=True)
class Snapshot:
    ok: bool
    status_code: int
    payload: dict[str, Any]


@dataclass(slots=True)
class MetricsSnapshot:
    raw_text: str
    values: dict[str, float]


@dataclass(slots=True)
class LoadTestStageResult:
    workload_class: str
    concurrency: int
    warmup_requests: int
    request_count: int
    success_count: int
    service_busy_count: int
    timeout_count: int
    upstream_error_count: int
    client_timeout_count: int
    connection_error_count: int
    unexpected_client_error_count: int
    malformed_json_count: int
    invalid_response_count: int
    unexpected_status_count: int
    status_mismatch_count: int
    http_status_counts: dict[str, int]
    error_code_counts: dict[str, int]
    http_outcome_counts: dict[str, int]
    status_counts: dict[str, int]
    client_latency_p50_ms: float
    client_latency_p95_ms: float
    client_latency_p99_ms: float
    client_latency_max_ms: float
    server_total_p50_ms: float | None
    server_total_p95_ms: float | None
    server_total_p99_ms: float | None
    server_deepseek_p50_ms: float | None
    server_deepseek_p95_ms: float | None
    server_deepseek_p99_ms: float | None
    server_embedding_p50_ms: float | None
    server_embedding_p95_ms: float | None
    server_embedding_p99_ms: float | None
    server_qdrant_p50_ms: float | None
    server_qdrant_p95_ms: float | None
    server_qdrant_p99_ms: float | None
    server_ranking_p50_ms: float | None
    server_ranking_p95_ms: float | None
    server_ranking_p99_ms: float | None
    external_overhead_p50_ms: float | None
    external_overhead_p95_ms: float | None
    external_overhead_p99_ms: float | None
    wall_seconds: float
    throughput_rps: float
    successful_rps: float
    service_busy_rate: float
    degraded: bool
    likely_bottleneck: str | None
    readiness_before: Snapshot
    readiness_after: Snapshot | None
    metrics_before: MetricsSnapshot
    metrics_after: MetricsSnapshot | None
    observations: list[LoadTestObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoadTestRunResult:
    git_commit: str | None
    git_dirty: bool
    started_at: str
    ended_at: str
    base_url: str
    endpoint: str
    dataset_path: str
    dataset_sha256: str
    qdrant_alias: str | None
    resolved_collection: str | None
    embedding_model: str | None
    deepseek_model: str | None
    max_concurrent_searches: int | None
    concurrency_levels: list[int]
    workloads: list[str]
    warmup_requests: int
    requests_per_stage: int
    limit: int
    client_timeout_seconds: float
    host_metrics_collected: bool
    host_metrics: dict[str, Any]
    preflight_passed: bool
    preflight_error_code_counts: dict[str, int]
    preflight_http_outcome_counts: dict[str, int]
    preflight_observations: list[LoadTestObservation]
    environment_status: str
    stages: list[LoadTestStageResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return True
    return bool(completed.stdout.strip())


def _dataset_sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cases.append(EvalCase.model_validate_json(line))
    if not cases:
        raise RuntimeError(f"No eval cases found in {path}")
    return cases


def _case_brand(case: EvalCase) -> str | None:
    return case.expected_attributes.brand


def _select_workload_cases(cases: list[EvalCase], workload_class: str) -> list[EvalCase]:
    if workload_class == "full_pipeline":
        return [
            case
            for case in cases
            if case.expected_status in {"exact_match", "not_found"}
            and _case_brand(case) is not None
        ]
    if workload_class == "article_fast_path":
        return [
            case
            for case in cases
            if case.expected_status in {"exact_match", "not_found"} and "article" in case.category
        ]
    if workload_class == "business_fast_fail":
        return [
            case
            for case in cases
            if case.expected_status == "cannot_process"
            and (
                case.category in {"unsupported_brand", "dn_semantic", "pn_semantic"}
                or "semantic" in case.category
                or "brand" in case.category
            )
        ]
    if workload_class == "mixed":
        full_pipeline = _select_workload_cases(cases, "full_pipeline")
        article_fast_path = _select_workload_cases(cases, "article_fast_path")
        business_fast_fail = _select_workload_cases(cases, "business_fast_fail")
        if not full_pipeline and not article_fast_path and not business_fast_fail:
            raise RuntimeError("No cases available for mixed workload")
        schedule: list[EvalCase] = []
        plan = [
            (full_pipeline, 8),
            (article_fast_path, 1),
            (business_fast_fail, 1),
        ]
        for pool, count in plan:
            if not pool:
                continue
            pool_cycle = cycle(pool)
            schedule.extend(next(pool_cycle) for _ in range(count))
        return schedule
    raise ValueError(f"Unknown workload class: {workload_class}")


def _select_load_cases(cases: list[EvalCase]) -> list[EvalCase]:
    return _select_workload_cases(cases, "full_pipeline")


def _build_request_schedule(
    cases: list[EvalCase],
    workload_class_or_request_count: str | int,
    request_count: int | None = None,
) -> list[EvalCase]:
    if request_count is None:
        workload_class = "full_pipeline"
        request_count = int(workload_class_or_request_count)
    else:
        workload_class = str(workload_class_or_request_count)
    pool = _select_workload_cases(cases, workload_class)
    if not pool:
        raise RuntimeError(f"No cases available for workload {workload_class}")
    case_cycle = cycle(pool)
    return [next(case_cycle) for _ in range(request_count)]


def _parse_csv_ints(value: str | list[str]) -> list[int]:
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]
    result: list[int] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    if not result:
        raise ValueError("At least one concurrency level is required")
    return result


def _parse_csv_strings(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]
    result: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                result.append(part)
    if not result:
        raise ValueError("At least one workload is required")
    return result


def _extract_request_id(response: httpx.Response, payload: dict[str, Any]) -> str | None:
    header_request_id = response.headers.get(REQUEST_ID_HEADER)
    if header_request_id:
        stripped = header_request_id.strip()
        if stripped:
            return stripped
    request_id = payload.get("request_id")
    if request_id is None:
        return None
    stripped = str(request_id).strip()
    return stripped or None


def _extract_server_timing(payload: dict[str, Any]) -> dict[str, float] | None:
    raw = payload.get("timing_ms")
    if not isinstance(raw, dict):
        return None
    timing: dict[str, float] = {}
    for key in ("deepseek", "embedding", "qdrant", "ranking", "total"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            timing[key] = float(value)
    return timing or None


def _classify_http_outcome(
    *,
    status_code: int,
    error_code: str | None,
    malformed_json: bool,
) -> str:
    if malformed_json:
        return "malformed_json"
    if error_code == "SERVICE_BUSY":
        return "service_busy"
    if error_code == "CLIENT_TIMEOUT":
        return "client_timeout"
    if error_code == "DEEPSEEK_TIMEOUT":
        return "deepseek_timeout"
    if error_code == "DEEPSEEK_UNAVAILABLE":
        return "deepseek_unavailable"
    if error_code == "EMBEDDING_TIMEOUT":
        return "embedding_timeout"
    if error_code == "EMBEDDING_UNAVAILABLE":
        return "embedding_unavailable"
    if error_code == "SEARCH_BACKEND_TIMEOUT":
        return "search_backend_timeout"
    if error_code == "SEARCH_BACKEND_UNAVAILABLE":
        return "search_backend_unavailable"
    if status_code == 429:
        return "http_429"
    if status_code == 500:
        return "http_500"
    if status_code == 502:
        return "http_502"
    if status_code == 503:
        return "http_503"
    if status_code == 504:
        return "http_504"
    if 400 <= status_code < 500:
        return f"http_{status_code}"
    if status_code >= 500:
        return f"http_{status_code}"
    return "success"


def _metrics_snapshot(text: str) -> MetricsSnapshot:
    import re

    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{.*\}\s+([+-]?[0-9]+(?:\.[0-9]+)?)$", line)
        if match is None:
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([+-]?[0-9]+(?:\.[0-9]+)?)$", line)
        if match is None:
            continue
        metric_name = match.group(1)
        values[metric_name] = values.get(metric_name, 0.0) + float(match.group(2))
    return MetricsSnapshot(raw_text=text, values=values)


async def _get_json(client: httpx.AsyncClient, path: str) -> Snapshot:
    response = await client.get(path)
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return Snapshot(
        ok=response.status_code == 200, status_code=response.status_code, payload=payload
    )


async def _get_metrics(client: httpx.AsyncClient) -> MetricsSnapshot:
    response = await client.get("/metrics")
    return _metrics_snapshot(response.text)


async def _perform_request(
    client: httpx.AsyncClient,
    case: EvalCase,
    *,
    request_number: int,
    concurrency: int,
    workload_class: str,
    limit: int,
    endpoint: str,
) -> LoadTestObservation:
    started = perf_counter()
    try:
        response = await client.post(endpoint, json={"query": case.query, "limit": limit})
    except httpx.TimeoutException as exc:
        return LoadTestObservation(
            request_number=request_number,
            concurrency=concurrency,
            workload_class=workload_class,
            query_id=case.id,
            category=case.category,
            expected_status=case.expected_status,
            query=case.query,
            status_code=504,
            http_outcome="client_timeout",
            latency_ms=(perf_counter() - started) * 1000.0,
            error_code="CLIENT_TIMEOUT",
            error_message=str(exc),
        )
    except httpx.ConnectError as exc:
        return LoadTestObservation(
            request_number=request_number,
            concurrency=concurrency,
            workload_class=workload_class,
            query_id=case.id,
            category=case.category,
            expected_status=case.expected_status,
            query=case.query,
            status_code=0,
            http_outcome="connection_error",
            latency_ms=(perf_counter() - started) * 1000.0,
            error_code="CONNECTION_ERROR",
            error_message=str(exc),
        )
    except httpx.HTTPError as exc:
        return LoadTestObservation(
            request_number=request_number,
            concurrency=concurrency,
            workload_class=workload_class,
            query_id=case.id,
            category=case.category,
            expected_status=case.expected_status,
            query=case.query,
            status_code=0,
            http_outcome="unexpected_client_error",
            latency_ms=(perf_counter() - started) * 1000.0,
            error_code="UNEXPECTED_CLIENT_ERROR",
            error_message=str(exc),
        )

    latency_ms = (perf_counter() - started) * 1000.0
    payload: dict[str, Any]
    malformed_json = False
    try:
        raw_payload = response.json()
        payload = raw_payload if isinstance(raw_payload, dict) else {"value": raw_payload}
    except Exception:
        payload = {}
        malformed_json = True

    request_id = _extract_request_id(response, payload)
    result_status = None
    server_timing = None
    error_code = None
    error_message = None
    if response.status_code == 200 and not malformed_json:
        result_status = payload.get("status")
        if result_status is not None:
            result_status = str(result_status)
        server_timing = _extract_server_timing(payload)
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            error_code = str(error.get("code") or "") or None
            error_message = str(error.get("message") or "") or None

    invalid_response = response.status_code == 200 and (
        malformed_json or result_status not in ALLOWED_RESULT_STATUSES
    )
    status_mismatch = bool(
        response.status_code == 200
        and result_status is not None
        and result_status != case.expected_status
    )
    http_outcome = _classify_http_outcome(
        status_code=response.status_code,
        error_code=error_code,
        malformed_json=malformed_json,
    )
    if response.status_code == 200 and not invalid_response:
        http_outcome = "success"
    if malformed_json:
        error_message = error_message or response.text[:200]

    return LoadTestObservation(
        request_number=request_number,
        concurrency=concurrency,
        workload_class=workload_class,
        query_id=case.id,
        category=case.category,
        expected_status=case.expected_status,
        query=case.query,
        status_code=response.status_code,
        http_outcome=http_outcome,
        result_status=result_status,
        latency_ms=latency_ms,
        server_timing_ms=server_timing,
        request_id=request_id,
        error_code=error_code,
        error_message=error_message,
        malformed_json=malformed_json,
        invalid_response=invalid_response,
        status_mismatch=status_mismatch,
    )


async def _run_requests(
    client: httpx.AsyncClient,
    schedule: list[EvalCase],
    *,
    concurrency: int,
    limit: int,
    endpoint: str,
    workload_class: str,
) -> tuple[list[LoadTestObservation], float]:
    queue: asyncio.Queue[tuple[int, EvalCase] | None] = asyncio.Queue()
    for request_number, case in enumerate(schedule, start=1):
        queue.put_nowait((request_number, case))
    for _ in range(concurrency):
        queue.put_nowait(None)

    observations: list[LoadTestObservation] = []
    observations_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            request_number, case = item
            observation = await _perform_request(
                client,
                case,
                request_number=request_number,
                concurrency=concurrency,
                workload_class=workload_class,
                limit=limit,
                endpoint=endpoint,
            )
            async with observations_lock:
                observations.append(observation)

    started = perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    return observations, perf_counter() - started


def _bottleneck_from_p95(stage: LoadTestStageResult) -> str | None:
    timings = {
        "DeepSeek": stage.server_deepseek_p95_ms,
        "Embedding": stage.server_embedding_p95_ms,
        "Qdrant": stage.server_qdrant_p95_ms,
        "Ranking": stage.server_ranking_p95_ms,
    }
    available = {name: value for name, value in timings.items() if value is not None}
    if not available:
        return None
    return max(available.items(), key=lambda item: item[1])[0]


def _stage_error_total(stage: LoadTestStageResult) -> int:
    return (
        stage.timeout_count
        + stage.upstream_error_count
        + stage.client_timeout_count
        + stage.connection_error_count
        + stage.unexpected_client_error_count
        + stage.malformed_json_count
        + stage.invalid_response_count
    )


def _summarize_stage(
    *,
    workload_class: str,
    concurrency: int,
    warmup_requests: int,
    observations: list[LoadTestObservation],
    wall_seconds: float,
    readiness_before: Snapshot,
    readiness_after: Snapshot | None,
    metrics_before: MetricsSnapshot,
    metrics_after: MetricsSnapshot | None,
) -> LoadTestStageResult:
    client_latencies = [observation.latency_ms for observation in observations]
    success_observations = [
        observation for observation in observations if observation.status_code == 200
    ]
    service_busy = [
        observation for observation in observations if observation.http_outcome == "service_busy"
    ]
    server_timeout = [
        observation
        for observation in observations
        if observation.http_outcome
        in {"deepseek_timeout", "embedding_timeout", "search_backend_timeout"}
    ]
    upstream_error = [
        observation
        for observation in observations
        if observation.http_outcome
        in {
            "deepseek_unavailable",
            "embedding_unavailable",
            "search_backend_unavailable",
        }
    ]
    client_timeout = [
        observation for observation in observations if observation.http_outcome == "client_timeout"
    ]
    connection_error = [
        observation
        for observation in observations
        if observation.http_outcome == "connection_error"
    ]
    unexpected_client_error = [
        observation
        for observation in observations
        if observation.http_outcome == "unexpected_client_error"
    ]
    malformed_json = [observation for observation in observations if observation.malformed_json]
    invalid_response = [observation for observation in observations if observation.invalid_response]
    status_mismatch = [observation for observation in observations if observation.status_mismatch]
    unexpected_status = [
        observation
        for observation in observations
        if observation.status_code == 200
        and observation.result_status not in ALLOWED_RESULT_STATUSES
    ]

    http_status_counts = Counter(f"HTTP_{observation.status_code}" for observation in observations)
    error_code_counts = Counter(
        observation.error_code or observation.http_outcome
        for observation in observations
        if observation.error_code or observation.http_outcome != "success"
    )
    http_outcome_counts = Counter(observation.http_outcome for observation in observations)
    status_counts = Counter(
        observation.result_status or f"HTTP_{observation.status_code}"
        for observation in observations
    )

    def _timing_series(key: str) -> list[float]:
        series: list[float] = []
        for observation in success_observations:
            if observation.server_timing_ms and key in observation.server_timing_ms:
                series.append(observation.server_timing_ms[key])
        return series

    server_total_series = _timing_series("total")
    server_deepseek_series = _timing_series("deepseek")
    server_embedding_series = _timing_series("embedding")
    server_qdrant_series = _timing_series("qdrant")
    server_ranking_series = _timing_series("ranking")
    external_overhead_series = [
        observation.latency_ms - observation.server_timing_ms["total"]
        for observation in success_observations
        if observation.server_timing_ms and observation.server_timing_ms.get("total") is not None
    ]

    stage = LoadTestStageResult(
        workload_class=workload_class,
        concurrency=concurrency,
        warmup_requests=warmup_requests,
        request_count=len(observations),
        success_count=len(success_observations),
        service_busy_count=len(service_busy),
        timeout_count=len(server_timeout),
        upstream_error_count=len(upstream_error),
        client_timeout_count=len(client_timeout),
        connection_error_count=len(connection_error),
        unexpected_client_error_count=len(unexpected_client_error),
        malformed_json_count=len(malformed_json),
        invalid_response_count=len(invalid_response),
        unexpected_status_count=len(unexpected_status),
        status_mismatch_count=len(status_mismatch),
        http_status_counts=dict(sorted(http_status_counts.items())),
        error_code_counts=dict(sorted(error_code_counts.items())),
        http_outcome_counts=dict(sorted(http_outcome_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
        client_latency_p50_ms=_percentile(client_latencies, 50) or 0.0,
        client_latency_p95_ms=_percentile(client_latencies, 95) or 0.0,
        client_latency_p99_ms=_percentile(client_latencies, 99) or 0.0,
        client_latency_max_ms=max(client_latencies) if client_latencies else 0.0,
        server_total_p50_ms=_percentile(server_total_series, 50),
        server_total_p95_ms=_percentile(server_total_series, 95),
        server_total_p99_ms=_percentile(server_total_series, 99),
        server_deepseek_p50_ms=_percentile(server_deepseek_series, 50),
        server_deepseek_p95_ms=_percentile(server_deepseek_series, 95),
        server_deepseek_p99_ms=_percentile(server_deepseek_series, 99),
        server_embedding_p50_ms=_percentile(server_embedding_series, 50),
        server_embedding_p95_ms=_percentile(server_embedding_series, 95),
        server_embedding_p99_ms=_percentile(server_embedding_series, 99),
        server_qdrant_p50_ms=_percentile(server_qdrant_series, 50),
        server_qdrant_p95_ms=_percentile(server_qdrant_series, 95),
        server_qdrant_p99_ms=_percentile(server_qdrant_series, 99),
        server_ranking_p50_ms=_percentile(server_ranking_series, 50),
        server_ranking_p95_ms=_percentile(server_ranking_series, 95),
        server_ranking_p99_ms=_percentile(server_ranking_series, 99),
        external_overhead_p50_ms=_percentile(external_overhead_series, 50),
        external_overhead_p95_ms=_percentile(external_overhead_series, 95),
        external_overhead_p99_ms=_percentile(external_overhead_series, 99),
        wall_seconds=wall_seconds,
        throughput_rps=_safe_div(len(observations), wall_seconds),
        successful_rps=_safe_div(len(success_observations), wall_seconds),
        service_busy_rate=_safe_div(len(service_busy), len(observations)),
        degraded=(
            _safe_div(len(service_busy), len(observations)) > 0
            or _safe_div(len(upstream_error) + len(server_timeout), len(observations)) > 0.01
        ),
        likely_bottleneck=None,
        readiness_before=readiness_before,
        readiness_after=readiness_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        observations=observations,
    )
    stage.likely_bottleneck = _bottleneck_from_p95(stage)
    return stage


async def _run_stage(
    *,
    client: httpx.AsyncClient,
    cases: list[EvalCase],
    workload_class: str,
    concurrency: int,
    requests_per_stage: int,
    warmup_requests: int,
    limit: int,
    endpoint: str,
) -> LoadTestStageResult | None:
    readiness_before = await _get_json(client, "/health/ready")
    if not readiness_before.ok:
        return None

    metrics_before = await _get_metrics(client)
    request_count = max(requests_per_stage, concurrency)

    warmup_schedule = _build_request_schedule(cases, workload_class, warmup_requests)
    if warmup_schedule:
        await _run_requests(
            client,
            warmup_schedule,
            concurrency=concurrency,
            limit=limit,
            endpoint=endpoint,
            workload_class=workload_class,
        )

    schedule = _build_request_schedule(cases, workload_class, request_count)
    observations, wall_seconds = await _run_requests(
        client,
        schedule,
        concurrency=concurrency,
        limit=limit,
        endpoint=endpoint,
        workload_class=workload_class,
    )

    readiness_after = await _get_json(client, "/health/ready")
    metrics_after = await _get_metrics(client)
    return _summarize_stage(
        workload_class=workload_class,
        concurrency=concurrency,
        warmup_requests=warmup_requests,
        observations=observations,
        wall_seconds=wall_seconds,
        readiness_before=readiness_before,
        readiness_after=readiness_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )


async def _run_preflight(
    *,
    client: httpx.AsyncClient,
    cases: list[EvalCase],
    limit: int,
    endpoint: str,
) -> tuple[bool, list[LoadTestObservation]]:
    schedule = _build_request_schedule(cases, "full_pipeline", PREFLIGHT_REQUEST_COUNT)
    observations: list[LoadTestObservation] = []
    for request_number, case in enumerate(schedule, start=1):
        observation = await _perform_request(
            client,
            case,
            request_number=request_number,
            concurrency=1,
            workload_class="preflight",
            limit=limit,
            endpoint=endpoint,
        )
        observations.append(observation)
        if observation.status_code != 200 or observation.http_outcome != "success":
            return False, observations
    return True, observations


def _collect_host_metrics() -> tuple[bool, dict[str, Any]]:
    try:
        import psutil  # type: ignore
    except Exception:
        return False, {"collected": False, "reason": "psutil_unavailable"}

    process = psutil.Process(os.getpid())
    with process.oneshot():
        return True, {
            "collected": True,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "rss_bytes": process.memory_info().rss,
        }


def _smoke_query(cases: list[EvalCase]) -> EvalCase:
    for case in cases:
        if case.expected_status == "exact_match" and _case_brand(case) is not None:
            return case
    return cases[0]


async def _run_load_test(
    *,
    dataset_path: Path,
    base_url: str | None,
    endpoint: str,
    concurrency_levels: list[int],
    workloads: list[str],
    requests_per_stage: int,
    warmup_requests: int,
    limit: int,
    client_timeout_seconds: float,
) -> LoadTestRunResult:
    cases = _load_cases(dataset_path)
    started_at = datetime.now(timezone.utc).isoformat()
    host_metrics_collected, host_metrics = _collect_host_metrics()
    settings = get_settings()

    if base_url is None:
        transport = httpx.ASGITransport(app=main.app)
        client_kwargs: dict[str, Any] = {
            "transport": transport,
            "base_url": "http://testserver",
            "trust_env": False,
        }
        context = main.app.router.lifespan_context(main.app)
        effective_base_url = "asgi://main.app"
    else:
        client_kwargs = {"base_url": base_url, "trust_env": False}
        context = None
        effective_base_url = base_url

    stages: list[LoadTestStageResult] = []
    timeout = httpx.Timeout(client_timeout_seconds)
    commit = _git_commit()
    dirty = _git_dirty()
    preflight_passed = False
    preflight_observations: list[LoadTestObservation] = []

    async with httpx.AsyncClient(timeout=timeout, **client_kwargs) as client:

        async def run_all_stages() -> None:
            nonlocal preflight_passed, preflight_observations

            live = await _get_json(client, "/health/live")
            ready = await _get_json(client, "/health/ready")
            if not live.ok or not ready.ok:
                preflight_passed = False
                preflight_observations = []
                return

            smoke_case = _smoke_query(cases)
            smoke = await _perform_request(
                client,
                smoke_case,
                request_number=0,
                concurrency=1,
                workload_class="smoke",
                limit=limit,
                endpoint=endpoint,
            )
            if smoke.status_code != 200 or smoke.http_outcome != "success":
                preflight_passed = False
                preflight_observations = [smoke]
                return

            preflight_passed, preflight_observations = await _run_preflight(
                client=client,
                cases=cases,
                limit=limit,
                endpoint=endpoint,
            )
            if not preflight_passed:
                return

            for workload_class in workloads:
                workload_cases = _select_workload_cases(cases, workload_class)
                if not workload_cases:
                    continue
                for concurrency in concurrency_levels:
                    stage = await _run_stage(
                        client=client,
                        cases=workload_cases,
                        workload_class=workload_class,
                        concurrency=concurrency,
                        requests_per_stage=requests_per_stage,
                        warmup_requests=warmup_requests,
                        limit=limit,
                        endpoint=endpoint,
                    )
                    if stage is not None:
                        stages.append(stage)

        if context is None:
            await run_all_stages()
        else:
            async with context:
                await run_all_stages()

    ended_at = datetime.now(timezone.utc).isoformat()
    qdrant_ready = next(
        (stage.readiness_before.payload for stage in stages if stage.readiness_before.ok), {}
    )
    qdrant_alias = None
    resolved_collection = None
    embedding_model = settings.embedding_model
    deepseek_model = settings.deepseek_model
    max_concurrent_searches = settings.max_concurrent_searches
    details = qdrant_ready.get("details") if isinstance(qdrant_ready, dict) else {}
    if isinstance(details, dict):
        qdrant_alias = details.get("collection_alias") or qdrant_ready.get("collection_alias")
        resolved_collection = details.get("resolved_collection_name") or qdrant_ready.get(
            "resolved_collection_name"
        )
        embedding_model = details.get("runtime_model") or embedding_model
        deepseek_model = details.get("deepseek_model") or deepseek_model

    environment_status = "valid" if preflight_passed and not dirty else "invalid"
    if not preflight_passed:
        environment_status = "invalid"

    preflight_error_code_counts = Counter(
        observation.error_code or observation.http_outcome
        for observation in preflight_observations
        if observation.error_code or observation.http_outcome != "success"
    )
    preflight_http_outcome_counts = Counter(
        observation.http_outcome for observation in preflight_observations
    )

    return LoadTestRunResult(
        git_commit=commit,
        git_dirty=dirty,
        started_at=started_at,
        ended_at=ended_at,
        base_url=effective_base_url,
        endpoint=endpoint,
        dataset_path=str(dataset_path),
        dataset_sha256=_dataset_sha256(dataset_path),
        qdrant_alias=qdrant_alias,
        resolved_collection=resolved_collection,
        embedding_model=embedding_model,
        deepseek_model=deepseek_model,
        max_concurrent_searches=max_concurrent_searches,
        concurrency_levels=concurrency_levels,
        workloads=workloads,
        warmup_requests=warmup_requests,
        requests_per_stage=requests_per_stage,
        limit=limit,
        client_timeout_seconds=client_timeout_seconds,
        host_metrics_collected=host_metrics_collected,
        host_metrics=host_metrics,
        preflight_passed=preflight_passed,
        preflight_error_code_counts=dict(sorted(preflight_error_code_counts.items())),
        preflight_http_outcome_counts=dict(sorted(preflight_http_outcome_counts.items())),
        preflight_observations=preflight_observations,
        environment_status=environment_status,
        stages=stages,
    )


def _render_error_counts(result: LoadTestRunResult) -> str:
    return json.dumps(result.preflight_error_code_counts, ensure_ascii=False)


def _render_report(result: LoadTestRunResult) -> str:
    lines: list[str] = [
        "# V5 Load Test",
        "",
        "## Run",
        f"- git_commit: `{result.git_commit or 'unknown'}`",
        f"- git_dirty: `{result.git_dirty}`",
        f"- environment_status: `{result.environment_status}`",
        f"- preflight_passed: `{result.preflight_passed}`",
        f"- preflight_error_codes: `{_render_error_counts(result)}`",
        f"- started_at: `{result.started_at}`",
        f"- ended_at: `{result.ended_at}`",
        f"- base url: `{result.base_url}`",
        f"- endpoint: `{result.endpoint}`",
        f"- dataset: `{result.dataset_path}`",
        f"- dataset sha256: `{result.dataset_sha256}`",
        f"- qdrant alias: `{result.qdrant_alias or 'unknown'}`",
        f"- resolved collection: `{result.resolved_collection or 'unknown'}`",
        f"- embedding model: `{result.embedding_model or 'unknown'}`",
        f"- deepseek model: `{result.deepseek_model or 'unknown'}`",
        f"- max concurrent searches: `{result.max_concurrent_searches or 'unknown'}`",
        f"- concurrency levels: `{', '.join(str(level) for level in result.concurrency_levels)}`",
        f"- workloads: `{', '.join(result.workloads)}`",
        f"- warmup requests: `{result.warmup_requests}`",
        f"- requests per stage: `{result.requests_per_stage}`",
        f"- client timeout seconds: `{result.client_timeout_seconds}`",
        "",
        "## Preflight",
    ]
    if result.preflight_passed:
        lines.append("- status: `passed`")
    else:
        lines.append("- status: `failed`")
    lines.append(f"- observations: `{len(result.preflight_observations)}`")
    lines.append(
        f"- error codes: `{json.dumps(result.preflight_error_code_counts, ensure_ascii=False)}`"
    )
    lines.extend(["", "## Totals"])

    total_requests = sum(stage.request_count for stage in result.stages)
    total_success = sum(stage.success_count for stage in result.stages)
    total_busy = sum(stage.service_busy_count for stage in result.stages)
    total_errors = sum(_stage_error_total(stage) for stage in result.stages)
    total_wall = sum(stage.wall_seconds for stage in result.stages)
    lines.extend(
        [
            f"- total requests: `{total_requests}`",
            f"- success: `{total_success}`",
            f"- busy (503 SERVICE_BUSY): `{total_busy}`",
            f"- other errors: `{total_errors}`",
            f"- throughput rps: `{_safe_div(total_requests, total_wall):.2f}`",
            f"- successful rps: `{_safe_div(total_success, total_wall):.2f}`",
            "",
            "## Stages",
            "",
            (
                "| Workload | C | Requests | Success | Busy | Errors | RPS | "
                "Successful RPS | p50 | p95 | p99 | Bottleneck |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for stage in result.stages:
        lines.append(
            "| "
            f"{stage.workload_class} | {stage.concurrency} | {stage.request_count} | "
            f"{stage.success_count} | {stage.service_busy_count} | "
            f"{_stage_error_total(stage)} | {stage.throughput_rps:.2f} | "
            f"{stage.successful_rps:.2f} | {stage.client_latency_p50_ms:.1f} | "
            f"{stage.client_latency_p95_ms:.1f} | {stage.client_latency_p99_ms:.1f} | "
            f"{stage.likely_bottleneck or 'unknown'} |"
        )

    lines.extend(
        [
            "",
            "## Server Timings",
            "",
            (
                "| Workload | C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | "
                "Server total p95 | External overhead p95 |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage in result.stages:
        lines.append(
            "| "
            f"{stage.workload_class} | {stage.concurrency} | "
            f"{(stage.server_deepseek_p95_ms or 0.0):.1f} | "
            f"{(stage.server_embedding_p95_ms or 0.0):.1f} | "
            f"{(stage.server_qdrant_p95_ms or 0.0):.1f} | "
            f"{(stage.server_ranking_p95_ms or 0.0):.1f} | "
            f"{(stage.server_total_p95_ms or 0.0):.1f} | "
            f"{(stage.external_overhead_p95_ms or 0.0):.1f} |"
        )

    lines.extend(["", "## Error Codes"])
    for stage in result.stages:
        error_codes = json.dumps(stage.error_code_counts, ensure_ascii=False)
        lines.append(f"- {stage.workload_class} C={stage.concurrency}: {error_codes}")

    if not result.host_metrics_collected:
        lines.extend(["", "## Host Metrics", "- host CPU/RAM metrics not collected"])
    else:
        lines.extend(
            ["", "## Host Metrics", f"- {json.dumps(result.host_metrics, ensure_ascii=False)}"]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a V5 load/stress test.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--base-url", type=str, default=None, help="Optional HTTP base URL")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--concurrency",
        type=_parse_csv_ints,
        default=DEFAULT_CONCURRENCY_LEVELS,
        help="Comma-separated concurrency levels or repeated values",
    )
    parser.add_argument(
        "--workloads",
        type=_parse_csv_strings,
        default=DEFAULT_WORKLOADS,
        help="Comma-separated workload classes or repeated values",
    )
    parser.add_argument("--requests-per-stage", type=int, default=DEFAULT_REQUESTS_PER_STAGE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--client-timeout-seconds", type=float, default=DEFAULT_CLIENT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY_LIMIT,
        help="Safety cap for requested concurrency",
    )
    parser.add_argument(
        "--max-requests-per-stage",
        type=int,
        default=MAX_REQUESTS_PER_STAGE,
        help="Safety cap for requested stage size",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main_cli(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    concurrency_levels = (
        args.concurrency if isinstance(args.concurrency, list) else [args.concurrency]
    )
    workloads = args.workloads if isinstance(args.workloads, list) else [args.workloads]
    if any(level <= 0 for level in concurrency_levels):
        raise ValueError("Concurrency levels must be positive integers")
    if max(concurrency_levels) > args.max_concurrency:
        raise ValueError("Concurrency exceeds safety cap")
    if args.requests_per_stage <= 0 or args.requests_per_stage > args.max_requests_per_stage:
        raise ValueError("requests-per-stage exceeds safety cap")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if any(workload not in WORKLOADS for workload in workloads):
        raise ValueError(f"Workloads must be one of: {', '.join(WORKLOADS)}")

    result = asyncio.run(
        _run_load_test(
            dataset_path=args.dataset,
            base_url=args.base_url,
            endpoint=args.endpoint,
            concurrency_levels=concurrency_levels,
            workloads=workloads,
            requests_per_stage=args.requests_per_stage,
            warmup_requests=args.warmup,
            limit=args.limit,
            client_timeout_seconds=args.client_timeout_seconds,
        )
    )
    payload = result.to_dict()
    write_results_json(payload, args.output_json)
    report = _render_report(result)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
