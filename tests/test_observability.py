from __future__ import annotations

import json
import logging

import rag_steel.observability as observability


def _reset_metrics() -> None:
    for metric in observability._METRICS:
        metric.values.clear()


def test_record_api_error_increments_bounded_code_counter() -> None:
    _reset_metrics()

    observability.record_api_error("SERVICE_BUSY")
    observability.record_api_error("SERVICE_BUSY")
    observability.record_api_error("DEEPSEEK_TIMEOUT")

    rendered = observability.render_metrics()

    assert '# HELP rag_api_errors_total API errors by machine-readable error code.' in rendered
    assert 'rag_api_errors_total{code="SERVICE_BUSY"} 2' in rendered
    assert 'rag_api_errors_total{code="DEEPSEEK_TIMEOUT"} 1' in rendered
    assert "request_id=" not in rendered
    assert "query=" not in rendered
    assert "message=" not in rendered
    assert "exception=" not in rendered


def test_search_in_flight_metric_is_exposed() -> None:
    _reset_metrics()

    observability.inc_search_in_flight()
    observability.inc_search_in_flight()
    observability.dec_search_in_flight()

    rendered = observability.render_metrics()

    assert (
        "# HELP rag_search_requests_in_flight "
        "In-flight search requests occupying concurrency slots."
    ) in rendered
    assert "rag_search_requests_in_flight 1" in rendered


def test_json_logs_use_context_request_id_for_upstream_failures(
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")
    token = observability.set_request_id("request-123")
    try:
        observability.log_deepseek_upstream_failure(
            upstream="deepseek",
            error_type="timeout",
            status_code=None,
            request_url="https://example.invalid/chat/completions",
            retryable=True,
            attempt=1,
            exception_type="ReadTimeout",
        )
    finally:
        observability.reset_request_id(token)

    payload = json.loads(caplog.records[0].message)
    assert payload["event"] == "upstream_failure"
    assert payload["request_id"] == "request-123"


def test_search_trace_helper_respects_enabled_flag(caplog) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")

    observability.log_search_trace("started", enabled=False, query="hidden")
    assert caplog.records == []

    observability.log_search_trace("started", enabled=True, request_id="rid-1", query="visible")

    payload = json.loads(caplog.records[0].message)
    assert payload == {
        "event": "search_trace",
        "request_id": "rid-1",
        "stage": "started",
        "query": "visible",
    }
