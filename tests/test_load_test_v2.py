from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.load_test_v2 import (
    LoadTestObservation,
    _load_queries,
    _percentile,
    _summarize_level,
)


def test_percentile_handles_empty_and_singleton() -> None:
    assert _percentile([], 95) is None
    assert _percentile([42.0], 95) == 42.0


def test_percentile_interpolates() -> None:
    assert _percentile([10.0, 20.0, 30.0, 40.0], 50) == pytest.approx(25.0)


def test_summarize_level_counts_errors_and_timings() -> None:
    observations = [
        LoadTestObservation(
            query="Temper DN80 PN16",
            status_code=200,
            latency_ms=120.0,
            response_status="exact_match",
            timing_ms={"total": 110.0, "embedding": 10.0, "qdrant": 20.0, "ranking": 30.0},
        ),
        LoadTestObservation(
            query="Temper DN80 PN16",
            status_code=503,
            latency_ms=15.0,
            error_code="SERVICE_BUSY",
            error_message="busy",
        ),
        LoadTestObservation(
            query="Temper DN80 PN16",
            status_code=504,
            latency_ms=18.0,
            error_code="TIMEOUT",
            error_message="timeout",
        ),
        LoadTestObservation(
            query="Temper DN80 PN16",
            status_code=500,
            latency_ms=9.0,
            error_code="BROKEN",
            error_message="broken",
        ),
    ]

    summary = _summarize_level(10, observations, wall_ms=400.0)

    assert summary.concurrency == 10
    assert summary.request_count == 4
    assert summary.success_count == 1
    assert summary.busy_count == 1
    assert summary.timeout_count == 1
    assert summary.error_count == 1
    assert summary.rps == pytest.approx(10.0)
    assert summary.client_p50_ms == pytest.approx(16.5)
    assert summary.server_total_p50_ms == pytest.approx(110.0)
    assert summary.server_embedding_p50_ms == pytest.approx(10.0)
    assert summary.server_qdrant_p50_ms == pytest.approx(20.0)
    assert summary.server_ranking_p50_ms == pytest.approx(30.0)
    assert summary.status_counts["exact_match"] == 1
    assert summary.status_counts["HTTP_503"] == 1
    assert summary.status_counts["HTTP_504"] == 1
    assert summary.status_counts["HTTP_500"] == 1


def test_load_queries_reads_jsonl(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            [
                json.dumps({"query": "Temper DN80 PN16"}, ensure_ascii=False),
                json.dumps({"query": "Broen DN80 PN16"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    assert _load_queries(queries_path) == ["Temper DN80 PN16", "Broen DN80 PN16"]
