from __future__ import annotations

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
