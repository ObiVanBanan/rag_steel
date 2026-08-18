from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.load_test_v2 import (
    LoadTestObservation,
    MetricsSnapshot,
    Snapshot,
    _build_request_schedule,
    _classify_http_outcome,
    _load_cases,
    _percentile,
    _run_requests,
    _select_workload_cases,
    _summarize_stage,
)
from eval.v3_schema import EvalCase


def _case(case_id: str, *, expected_status: str, brand: str | None = "Temper") -> SimpleNamespace:
    return SimpleNamespace(
        id=case_id,
        category=f"category_{case_id}",
        query=f"query_{case_id}",
        expected_status=expected_status,
        expected_attributes=SimpleNamespace(brand=brand),
    )


def _observation(
    *,
    request_number: int,
    status_code: int,
    http_outcome: str,
    latency_ms: float,
    result_status: str | None = None,
    error_code: str | None = None,
    malformed_json: bool = False,
    invalid_response: bool = False,
    status_mismatch: bool = False,
    server_timing_ms: dict[str, float] | None = None,
) -> LoadTestObservation:
    return LoadTestObservation(
        request_number=request_number,
        concurrency=10,
        workload_class="full_pipeline",
        query_id=f"case-{request_number}",
        category="brand_dn_pn",
        expected_status="exact_match",
        query=f"query-{request_number}",
        status_code=status_code,
        http_outcome=http_outcome,
        result_status=result_status,
        latency_ms=latency_ms,
        server_timing_ms=server_timing_ms,
        request_id=f"req-{request_number}",
        error_code=error_code,
        malformed_json=malformed_json,
        invalid_response=invalid_response,
        status_mismatch=status_mismatch,
    )


def test_percentile_handles_empty_and_interpolates() -> None:
    assert _percentile([], 95) is None
    assert _percentile([42.0], 95) == 42.0
    assert _percentile([10.0, 20.0, 30.0, 40.0], 50) == pytest.approx(25.0)


def test_classify_http_outcome_covers_key_failures() -> None:
    assert (
        _classify_http_outcome(status_code=503, error_code="SERVICE_BUSY", malformed_json=False)
        == "service_busy"
    )
    assert (
        _classify_http_outcome(status_code=504, error_code="DEEPSEEK_TIMEOUT", malformed_json=False)
        == "deepseek_timeout"
    )
    assert (
        _classify_http_outcome(status_code=504, error_code=None, malformed_json=False) == "http_504"
    )
    assert (
        _classify_http_outcome(status_code=0, error_code=None, malformed_json=True)
        == "malformed_json"
    )


def test_load_cases_reads_eval_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "brand_dn_pn",
                "query": "Temper DN80 PN16",
                "expected_status": "exact_match",
                "expected_attributes": {"brand": "Temper"},
                "eligible_competitor_articles": [],
                "preferred_competitor_articles": [],
                "expected_ld_articles_by_competitor": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = _load_cases(dataset)
    assert len(cases) == 1
    assert isinstance(cases[0], EvalCase)
    assert cases[0].id == "case-1"


def test_workload_selection_and_mixed_schedule() -> None:
    cases = [
        _case("1", expected_status="exact_match", brand="Temper"),
        _case("2", expected_status="not_found", brand="Broen"),
        _case("3", expected_status="cannot_process", brand=None),
        _case("4", expected_status="exact_match", brand="ALSO"),
    ]

    full_pipeline = _select_workload_cases(cases, "full_pipeline")
    brand_gate = _select_workload_cases(cases, "brand_gate_fast_path")
    mixed = _select_workload_cases(cases, "mixed")

    assert [case.id for case in full_pipeline] == ["1", "2", "4"]
    assert [case.id for case in brand_gate] == ["3"]
    assert len(mixed) == 10
    assert [case.expected_status for case in mixed[:10]] == [
        "exact_match",
        "exact_match",
        "exact_match",
        "exact_match",
        "exact_match",
        "exact_match",
        "exact_match",
        "exact_match",
        "not_found",
        "cannot_process",
    ]

    schedule = _build_request_schedule(cases, "full_pipeline", 5)
    assert [case.id for case in schedule] == ["1", "2", "4", "1", "2"]


def test_summarize_stage_counts_errors_timings_and_bottleneck() -> None:
    observations = [
        _observation(
            request_number=1,
            status_code=200,
            http_outcome="success",
            latency_ms=120.0,
            result_status="exact_match",
            server_timing_ms={
                "total": 110.0,
                "deepseek": 70.0,
                "embedding": 10.0,
                "qdrant": 20.0,
                "ranking": 5.0,
            },
        ),
        _observation(
            request_number=2,
            status_code=200,
            http_outcome="success",
            latency_ms=90.0,
            result_status="weird",
            invalid_response=True,
            status_mismatch=True,
            server_timing_ms={
                "total": 80.0,
                "deepseek": 60.0,
                "embedding": 8.0,
                "qdrant": 7.0,
                "ranking": 2.0,
            },
        ),
        _observation(
            request_number=3,
            status_code=503,
            http_outcome="service_busy",
            latency_ms=15.0,
            error_code="SERVICE_BUSY",
        ),
        _observation(
            request_number=4,
            status_code=504,
            http_outcome="client_timeout",
            latency_ms=18.0,
            error_code="CLIENT_TIMEOUT",
        ),
    ]

    stage = _summarize_stage(
        workload_class="full_pipeline",
        concurrency=10,
        warmup_requests=5,
        observations=observations,
        wall_seconds=1.0,
        readiness_before=Snapshot(ok=True, status_code=200, payload={"status": "ok"}),
        readiness_after=Snapshot(ok=True, status_code=200, payload={"status": "ok"}),
        metrics_before=MetricsSnapshot(
            raw_text="metrics-before", values={"rag_http_requests_total": 1.0}
        ),
        metrics_after=MetricsSnapshot(
            raw_text="metrics-after", values={"rag_http_requests_total": 2.0}
        ),
    )

    assert stage.request_count == 4
    assert stage.success_count == 2
    assert stage.service_busy_count == 1
    assert stage.timeout_count == 1
    assert stage.invalid_response_count == 1
    assert stage.status_mismatch_count == 1
    assert stage.http_status_counts["HTTP_503"] == 1
    assert stage.error_code_counts["SERVICE_BUSY"] == 1
    assert stage.http_outcome_counts["service_busy"] == 1
    assert stage.status_counts["exact_match"] == 1
    assert stage.status_counts["weird"] == 1
    assert stage.client_latency_max_ms == 120.0
    assert stage.server_total_p95_ms == pytest.approx(108.5)
    assert stage.server_deepseek_p95_ms == pytest.approx(69.5)
    assert stage.server_embedding_p95_ms == pytest.approx(9.9)
    assert stage.server_qdrant_p95_ms == pytest.approx(19.35)
    assert stage.server_ranking_p95_ms == pytest.approx(4.85)
    assert stage.external_overhead_p95_ms == pytest.approx(10.0)
    assert stage.service_busy_rate == pytest.approx(0.25)
    assert stage.degraded is True
    assert stage.likely_bottleneck == "DeepSeek"


def test_run_requests_is_bounded_by_worker_pool() -> None:
    cases = [_case(str(index), expected_status="exact_match") for index in range(5)]

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers = {"X-Request-ID": "req-1"}
            self.text = json.dumps(payload)

        def json(self) -> dict[str, object]:
            return self._payload

    class SlowClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return FakeResponse(
                200,
                {
                    "status": "exact_match",
                    "timing_ms": {
                        "total": 1.0,
                        "deepseek": 0.2,
                        "embedding": 0.2,
                        "qdrant": 0.3,
                        "ranking": 0.1,
                    },
                },
            )

    client = SlowClient()
    observations, wall_seconds = asyncio.run(
        _run_requests(
            client,  # type: ignore[arg-type]
            cases,
            concurrency=2,
            limit=5,
            endpoint="/v2/search",
            workload_class="full_pipeline",
        )
    )

    assert len(observations) == 5
    assert wall_seconds > 0
    assert client.max_active <= 2
