"""Run a production-style concurrency load test against the V5 search API."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any

import httpx

import main
from eval.load_test_v2 import (
    LoadTestRunResult,
    LoadTestStageResult,
    _collect_host_metrics,
    _dataset_sha256,
    _get_json,
    _get_metrics,
    _run_requests,
    _safe_div,
    _summarize_stage,
)
from eval.v5_constants import DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_RESULTS_DIR
from eval.v5_schema import EvalCase
from rag_steel.settings import get_settings

DEFAULT_REPORT_PATH = Path("eval/load_v5_report.md")
DEFAULT_RESULTS_JSON_PATH = DEFAULT_RESULTS_DIR / "load_v5_latest.json"
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 5, 10, 20, 50]
DEFAULT_REQUESTS_PER_STAGE = 100
DEFAULT_WARMUP_REQUESTS = 5
DEFAULT_LIMIT = 5
DEFAULT_CLIENT_TIMEOUT_SECONDS = 180.0
DEFAULT_ENDPOINT = "/v2/search"
MAX_CONCURRENCY_LIMIT = 100
MAX_REQUESTS_PER_STAGE = 1000
LOAD_POOL_SPECS: tuple[tuple[str, int], ...] = (
    ("brand_semantic", 2),
    ("brand_typo", 2),
    ("mixed_semantic", 2),
    ("dn_semantic", 2),
    ("pn_semantic", 2),
    ("connection_semantic", 2),
    ("ambiguous_semantic", 2),
    ("unsupported_brand", 1),
    ("real_user_regression", 3),
    ("article_only_exact", 1),
    ("article_only_typo", 1),
    ("brand_plus_article", 1),
    ("article_plus_hard", 1),
    ("article_natural_language", 1),
    ("unknown_article", 1),
    ("brand_article_conflict", 1),
)


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


def _select_load_cases(cases: list[EvalCase]) -> list[EvalCase]:
    selected: list[EvalCase] = []
    seen: set[str] = set()
    for category, count in LOAD_POOL_SPECS:
        matches = [case for case in cases if case.category == category]
        if not matches:
            continue
        for case in matches[:count]:
            if case.id in seen:
                continue
            selected.append(case)
            seen.add(case.id)
    if not selected:
        raise RuntimeError("No cases available for the V5 load baseline")
    return selected


def _build_request_schedule(cases: list[EvalCase], request_count: int) -> list[EvalCase]:
    pool = _select_load_cases(cases)
    case_cycle = cycle(pool)
    return [next(case_cycle) for _ in range(request_count)]


async def _run_stage(
    *,
    client: httpx.AsyncClient,
    cases: list[EvalCase],
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

    warmup_schedule = _build_request_schedule(cases, warmup_requests)
    if warmup_schedule:
        await _run_requests(
            client,
            warmup_schedule,
            concurrency=concurrency,
            limit=limit,
            endpoint=endpoint,
            workload_class="baseline",
        )

    schedule = _build_request_schedule(cases, request_count)
    observations, wall_seconds = await _run_requests(
        client,
        schedule,
        concurrency=concurrency,
        limit=limit,
        endpoint=endpoint,
        workload_class="baseline",
    )

    readiness_after = await _get_json(client, "/health/ready")
    metrics_after = await _get_metrics(client)
    return _summarize_stage(
        workload_class="baseline",
        concurrency=concurrency,
        warmup_requests=warmup_requests,
        observations=observations,
        wall_seconds=wall_seconds,
        readiness_before=readiness_before,
        readiness_after=readiness_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )


def _render_report(result: LoadTestRunResult) -> str:
    lines: list[str] = [
        "# V5 Load Test",
        "",
        "## Run",
        f"- commit: `{result.git_commit or 'unknown'}`",
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
        f"- warmup requests: `{result.warmup_requests}`",
        f"- requests per stage: `{result.requests_per_stage}`",
        f"- client timeout seconds: `{result.client_timeout_seconds}`",
        "",
        "## Totals",
    ]
    total_requests = sum(stage.request_count for stage in result.stages)
    total_success = sum(stage.success_count for stage in result.stages)
    total_busy = sum(stage.service_busy_count for stage in result.stages)
    total_errors = sum(
        stage.timeout_count
        + stage.upstream_error_count
        + stage.client_timeout_count
        + stage.connection_error_count
        + stage.unexpected_client_error_count
        + stage.invalid_response_count
        for stage in result.stages
    )
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
                "| C | Requests | Success | Busy | Errors | RPS | "
                "Successful RPS | p50 | p95 | p99 | Bottleneck |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for stage in result.stages:
        errors = (
            stage.timeout_count
            + stage.upstream_error_count
            + stage.client_timeout_count
            + stage.connection_error_count
            + stage.unexpected_client_error_count
            + stage.invalid_response_count
        )
        lines.append(
            "| "
            f"{stage.concurrency} | {stage.request_count} | {stage.success_count} | "
            f"{stage.service_busy_count} | {errors} | {stage.throughput_rps:.2f} | "
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
                "| C | DeepSeek p95 | Embed p95 | Qdrant p95 | Ranking p95 | "
                "Server total p95 | External overhead p95 |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage in result.stages:
        lines.append(
            "| "
            f"{stage.concurrency} | "
            f"{(stage.server_deepseek_p95_ms or 0.0):.1f} | "
            f"{(stage.server_embedding_p95_ms or 0.0):.1f} | "
            f"{(stage.server_qdrant_p95_ms or 0.0):.1f} | "
            f"{(stage.server_ranking_p95_ms or 0.0):.1f} | "
            f"{(stage.server_total_p95_ms or 0.0):.1f} | "
            f"{(stage.external_overhead_p95_ms or 0.0):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Saturation",
        ]
    )
    for stage in result.stages:
        timeout_total = (
            stage.timeout_count
            + stage.upstream_error_count
            + stage.client_timeout_count
            + stage.connection_error_count
            + stage.unexpected_client_error_count
        )
        readiness_after = stage.readiness_after.status_code if stage.readiness_after else "n/a"
        lines.append(
            f"- C={stage.concurrency}: busy={stage.service_busy_count}, "
            f"5xx+timeouts={timeout_total}, degraded={stage.degraded}, "
            f"readiness_before={stage.readiness_before.status_code}, "
            f"readiness_after={readiness_after}"
        )

    lines.extend(
        [
            "",
            "## Error Codes",
        ]
    )
    for stage in result.stages:
        lines.append(
            f"- C={stage.concurrency}: {json.dumps(stage.error_code_counts, ensure_ascii=False)}"
        )

    if not result.host_metrics_collected:
        lines.extend(["", "## Host Metrics", "- host CPU/RAM metrics not collected"])
    else:
        lines.extend(
            [
                "",
                "## Host Metrics",
                f"- {json.dumps(result.host_metrics, ensure_ascii=False)}",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


async def _run_load_test(
    *,
    dataset_path: Path,
    base_url: str | None,
    endpoint: str,
    concurrency_levels: list[int],
    requests_per_stage: int,
    warmup_requests: int,
    limit: int,
    client_timeout_seconds: float,
) -> LoadTestRunResult:
    cases = _load_cases(dataset_path)
    baseline_cases = _select_load_cases(cases)
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

    async with httpx.AsyncClient(timeout=timeout, **client_kwargs) as client:

        async def run_all_stages() -> None:
            for concurrency in concurrency_levels:
                stage = await _run_stage(
                    client=client,
                    cases=baseline_cases,
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
        (stage.readiness_before.payload for stage in stages if stage.readiness_before.ok),
        {},
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

    return LoadTestRunResult(
        git_commit=commit,
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
        workloads=["baseline"],
        warmup_requests=warmup_requests,
        requests_per_stage=requests_per_stage,
        limit=limit,
        client_timeout_seconds=client_timeout_seconds,
        host_metrics_collected=host_metrics_collected,
        host_metrics=host_metrics,
        stages=stages,
    )


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
        type=lambda value: [int(part) for part in str(value).split(",") if part.strip()],
        default=DEFAULT_CONCURRENCY_LEVELS,
        help="Comma-separated concurrency levels or repeated values",
    )
    parser.add_argument("--requests-per-stage", type=int, default=DEFAULT_REQUESTS_PER_STAGE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_REQUESTS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--client-timeout-seconds",
        type=float,
        default=DEFAULT_CLIENT_TIMEOUT_SECONDS,
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
    if any(level <= 0 for level in concurrency_levels):
        raise ValueError("Concurrency levels must be positive integers")
    if max(concurrency_levels) > args.max_concurrency:
        raise ValueError("Concurrency exceeds safety cap")
    if args.requests_per_stage <= 0 or args.requests_per_stage > args.max_requests_per_stage:
        raise ValueError("requests-per-stage exceeds safety cap")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    result = asyncio.run(
        _run_load_test(
            dataset_path=args.dataset,
            base_url=args.base_url,
            endpoint=args.endpoint,
            concurrency_levels=concurrency_levels,
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
