"""Request context, structured logging, and lightweight Prometheus metrics."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

REQUEST_ID_HEADER = "X-Request-ID"
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
DEFAULT_HISTOGRAM_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

_request_id_context: ContextVar[str | None] = ContextVar("rag_steel_request_id", default=None)
_logger = logging.getLogger("rag_steel.observability")


def get_request_id() -> str | None:
    return _request_id_context.get()


def set_request_id(request_id: str | None) -> object:
    return _request_id_context.set(request_id)


def reset_request_id(token: object) -> None:
    _request_id_context.reset(token)


def _safe_request_id(request_id: str | None = None) -> str | None:
    if request_id:
        stripped = request_id.strip()
        if stripped:
            return stripped
    return None


def resolve_request_id(header_value: str | None) -> str:
    from uuid import uuid4

    return _safe_request_id(header_value) or uuid4().hex


def _json_log(event: str, **payload: Any) -> None:
    request_id = payload.pop("request_id", None) or get_request_id()
    record = {"event": event, "request_id": request_id, **payload}
    _logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def log_search_trace(
    stage: str,
    *,
    enabled: bool,
    request_id: str | None = None,
    **payload: Any,
) -> None:
    if not enabled:
        return
    _json_log("search_trace", request_id=request_id, stage=stage, **payload)


def log_deepseek_upstream_failure(
    *,
    upstream: str,
    error_type: str,
    status_code: int | None,
    request_url: str | None,
    retryable: bool,
    attempt: int,
    exception_type: str,
) -> None:
    _json_log(
        "upstream_failure",
        upstream=upstream,
        error_type=error_type,
        status_code=status_code,
        request_url=request_url,
        retryable=retryable,
        attempt=attempt,
        exception_type=exception_type,
    )


def log_http_request_completed(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    _json_log(
        "http_request_completed",
        request_id=request_id,
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration_ms, 3),
    )


def log_batch_item_diagnostic(
    *,
    item_index: int,
    input_query: str,
    status: str,
    duration_ms: float,
    product_family: str | None = None,
    missing_fields: list[str] | None = None,
    results_count: int = 0,
    error_code: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "item_index": item_index,
        "input": input_query,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "results_count": results_count,
    }
    if product_family is not None:
        payload["product_family"] = product_family
    if missing_fields:
        payload["missing_fields"] = missing_fields
    if error_code is not None:
        payload["error_code"] = error_code
    _json_log("batch_item_diagnostic", **payload)


def log_batch_completed(
    *,
    batch_size: int,
    duration_ms: float,
    status_counts: dict[str, int],
) -> None:
    _json_log(
        "batch_completed",
        batch_size=batch_size,
        duration_ms=round(duration_ms, 3),
        status_counts=status_counts,
    )


def log_search_completed(
    *,
    request_id: str | None,
    result_status: str,
    results_count: int,
    total_ms: float,
    deepseek_ms: float | None = None,
    embedding_ms: float | None = None,
    qdrant_ms: float | None = None,
    ranking_ms: float | None = None,
    query_length: int | None = None,
    resolution_mode: str | None = None,
    error_type: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "result_status": result_status,
        "results_count": results_count,
        "total_ms": round(total_ms, 3),
    }
    if deepseek_ms is not None:
        payload["deepseek_ms"] = round(deepseek_ms, 3)
    if embedding_ms is not None:
        payload["embedding_ms"] = round(embedding_ms, 3)
    if qdrant_ms is not None:
        payload["qdrant_ms"] = round(qdrant_ms, 3)
    if ranking_ms is not None:
        payload["ranking_ms"] = round(ranking_ms, 3)
    if query_length is not None:
        payload["query_length"] = query_length
    if resolution_mode is not None:
        payload["resolution_mode"] = resolution_mode
    if error_type is not None:
        payload["error_type"] = error_type
    _json_log("search_completed", **payload)


def _normalize_label_names(label_names: tuple[str, ...], labels: dict[str, Any]) -> tuple[str, ...]:
    try:
        return tuple(str(labels[name]) for name in label_names)
    except KeyError as exc:  # pragma: no cover - guarded by callers
        missing = exc.args[0]
        raise KeyError(f"Missing metric label: {missing}") from exc


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_labels(label_names: tuple[str, ...], label_values: tuple[str, ...]) -> str:
    if not label_names:
        return ""
    parts = [
        f'{name}="{_escape_label_value(value)}"'
        for name, value in zip(label_names, label_values, strict=True)
    ]
    return "{" + ",".join(parts) + "}"


@dataclass(slots=True)
class CounterMetric:
    name: str
    help_text: str
    label_names: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        label_values = _normalize_label_names(self.label_names, labels)
        with self.lock:
            self.values[label_values] = self.values.get(label_values, 0.0) + value

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        for label_values, value in sorted(self.values.items()):
            lines.append(f"{self.name}{_format_labels(self.label_names, label_values)} {value:g}")
        if not self.values and not self.label_names:
            lines.append(f"{self.name} 0")
        return lines


@dataclass(slots=True)
class GaugeMetric:
    name: str
    help_text: str
    label_names: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)

    def set(self, value: float, **labels: Any) -> None:
        label_values = _normalize_label_names(self.label_names, labels)
        with self.lock:
            self.values[label_values] = value

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        label_values = _normalize_label_names(self.label_names, labels)
        with self.lock:
            self.values[label_values] = self.values.get(label_values, 0.0) + value

    def dec(self, value: float = 1.0, **labels: Any) -> None:
        self.inc(-value, **labels)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        for label_values, value in sorted(self.values.items()):
            lines.append(f"{self.name}{_format_labels(self.label_names, label_values)} {value:g}")
        if not self.values and not self.label_names:
            lines.append(f"{self.name} 0")
        return lines


@dataclass(slots=True)
class HistogramMetric:
    name: str
    help_text: str
    label_names: tuple[str, ...] = ()
    buckets: tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS
    values: dict[tuple[str, ...], dict[str, Any]] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)

    def observe(self, value: float, **labels: Any) -> None:
        label_values = _normalize_label_names(self.label_names, labels)
        with self.lock:
            series = self.values.setdefault(
                label_values,
                {
                    "bucket_counts": [0 for _ in self.buckets],
                    "count": 0,
                    "sum": 0.0,
                },
            )
            series["count"] += 1
            series["sum"] += value
            for index, bucket in enumerate(self.buckets):
                if value <= bucket:
                    series["bucket_counts"][index] += 1
                    break

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        for label_values, series in sorted(self.values.items()):
            cumulative = 0
            for bucket, bucket_count in zip(self.buckets, series["bucket_counts"], strict=True):
                cumulative += bucket_count
                lines.append(
                    f"{self.name}_bucket"
                    f"{_format_labels(self.label_names + ('le',), label_values + (f'{bucket:g}',))}"
                    f" {cumulative:g}"
                )
            lines.append(
                f"{self.name}_bucket"
                f"{_format_labels(self.label_names + ('le',), label_values + ('+Inf',))}"
                f" {series['count']:g}"
            )
            lines.append(
                f"{self.name}_sum{_format_labels(self.label_names, label_values)} {series['sum']:g}"
            )
            lines.append(
                f"{self.name}_count"
                f"{_format_labels(self.label_names, label_values)} {series['count']:g}"
            )
        if not self.values and not self.label_names:
            lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
            lines.append(f"{self.name}_sum 0")
            lines.append(f"{self.name}_count 0")
        return lines


HTTP_REQUESTS_TOTAL = CounterMetric(
    "rag_http_requests_total",
    "Total HTTP requests.",
    ("method", "path", "status_code"),
)
HTTP_REQUEST_DURATION = HistogramMetric(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "path"),
)
HTTP_REQUESTS_IN_FLIGHT = GaugeMetric(
    "rag_http_requests_in_flight",
    "In-flight HTTP requests.",
)
SEARCH_REQUESTS_IN_FLIGHT = GaugeMetric(
    "rag_search_requests_in_flight",
    "In-flight search requests occupying concurrency slots.",
)
SEARCH_REQUESTS_TOTAL = CounterMetric(
    "rag_search_requests_total",
    "Search requests by result status.",
    ("result_status",),
)
API_ERRORS_TOTAL = CounterMetric(
    "rag_api_errors_total",
    "API errors by machine-readable error code.",
    ("code",),
)
DEEPSEEK_REQUESTS_TOTAL = CounterMetric(
    "rag_deepseek_requests_total",
    "DeepSeek requests.",
)
DEEPSEEK_ERRORS_TOTAL = CounterMetric(
    "rag_deepseek_errors_total",
    "DeepSeek errors by type.",
    ("error_type",),
)
DEEPSEEK_DURATION = HistogramMetric(
    "rag_deepseek_duration_seconds",
    "DeepSeek request duration in seconds.",
)
EMBEDDING_REQUESTS_TOTAL = CounterMetric(
    "rag_embedding_requests_total",
    "Embedding requests.",
)
EMBEDDING_ERRORS_TOTAL = CounterMetric(
    "rag_embedding_errors_total",
    "Embedding errors by type.",
    ("error_type",),
)
EMBEDDING_DURATION = HistogramMetric(
    "rag_embedding_duration_seconds",
    "Embedding request duration in seconds.",
)
QDRANT_REQUESTS_TOTAL = CounterMetric(
    "rag_qdrant_requests_total",
    "Qdrant requests.",
)
QDRANT_ERRORS_TOTAL = CounterMetric(
    "rag_qdrant_errors_total",
    "Qdrant errors by type.",
    ("error_type",),
)
QDRANT_DURATION = HistogramMetric(
    "rag_qdrant_duration_seconds",
    "Qdrant request duration in seconds.",
)
RANKING_DURATION = HistogramMetric(
    "rag_ranking_duration_seconds",
    "Ranking duration in seconds.",
)
BATCH_SIZE = HistogramMetric(
    "rag_batch_size",
    "Number of products submitted in one v2 batch request.",
    buckets=(1, 2, 5, 10, 20, 50, 100),
)
BATCH_ITEMS_TOTAL = CounterMetric(
    "rag_batch_items_total",
    "Batch items by final status.",
    ("status",),
)
BATCH_ITEM_DURATION = HistogramMetric(
    "rag_batch_item_duration_seconds",
    "Per-item batch processing duration in seconds.",
    ("status",),
)
REQUIRED_PARAMETER_CHECKS_TOTAL = CounterMetric(
    "rag_required_parameter_checks_total",
    "Required-parameter checks by product family and result.",
    ("product_family", "result"),
)
REQUIRED_PARAMETER_MISSING_TOTAL = CounterMetric(
    "rag_required_parameter_missing_total",
    "Missing required parameters by product family and field.",
    ("product_family", "field"),
)

_METRICS = (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_IN_FLIGHT,
    SEARCH_REQUESTS_IN_FLIGHT,
    SEARCH_REQUESTS_TOTAL,
    API_ERRORS_TOTAL,
    DEEPSEEK_REQUESTS_TOTAL,
    DEEPSEEK_ERRORS_TOTAL,
    DEEPSEEK_DURATION,
    EMBEDDING_REQUESTS_TOTAL,
    EMBEDDING_ERRORS_TOTAL,
    EMBEDDING_DURATION,
    QDRANT_REQUESTS_TOTAL,
    QDRANT_ERRORS_TOTAL,
    QDRANT_DURATION,
    RANKING_DURATION,
    BATCH_SIZE,
    BATCH_ITEMS_TOTAL,
    BATCH_ITEM_DURATION,
    REQUIRED_PARAMETER_CHECKS_TOTAL,
    REQUIRED_PARAMETER_MISSING_TOTAL,
)


def record_http_request(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    HTTP_REQUESTS_TOTAL.inc(method=method, path=path, status_code=str(status_code))
    HTTP_REQUEST_DURATION.observe(duration_seconds, method=method, path=path)


def inc_in_flight() -> None:
    HTTP_REQUESTS_IN_FLIGHT.inc()


def dec_in_flight() -> None:
    HTTP_REQUESTS_IN_FLIGHT.dec()


def inc_search_in_flight() -> None:
    SEARCH_REQUESTS_IN_FLIGHT.inc()


def dec_search_in_flight() -> None:
    SEARCH_REQUESTS_IN_FLIGHT.dec()


def record_search_request(result_status: str) -> None:
    SEARCH_REQUESTS_TOTAL.inc(result_status=result_status)


def record_api_error(code: str) -> None:
    API_ERRORS_TOTAL.inc(code=code)


def record_deepseek_request(duration_seconds: float) -> None:
    DEEPSEEK_REQUESTS_TOTAL.inc()
    DEEPSEEK_DURATION.observe(duration_seconds)


def record_deepseek_error(error_type: str, duration_seconds: float) -> None:
    DEEPSEEK_REQUESTS_TOTAL.inc()
    DEEPSEEK_ERRORS_TOTAL.inc(error_type=error_type)
    DEEPSEEK_DURATION.observe(duration_seconds)


def record_embedding_request(duration_seconds: float) -> None:
    EMBEDDING_REQUESTS_TOTAL.inc()
    EMBEDDING_DURATION.observe(duration_seconds)


def record_embedding_error(error_type: str, duration_seconds: float) -> None:
    EMBEDDING_REQUESTS_TOTAL.inc()
    EMBEDDING_ERRORS_TOTAL.inc(error_type=error_type)
    EMBEDDING_DURATION.observe(duration_seconds)


def record_qdrant_request(duration_seconds: float) -> None:
    QDRANT_REQUESTS_TOTAL.inc()
    QDRANT_DURATION.observe(duration_seconds)


def record_qdrant_error(error_type: str, duration_seconds: float) -> None:
    QDRANT_REQUESTS_TOTAL.inc()
    QDRANT_ERRORS_TOTAL.inc(error_type=error_type)
    QDRANT_DURATION.observe(duration_seconds)


def record_ranking_duration(duration_seconds: float) -> None:
    RANKING_DURATION.observe(duration_seconds)


def record_batch_size(size: int) -> None:
    BATCH_SIZE.observe(float(size))


def record_batch_item(status: str, duration_seconds: float) -> None:
    BATCH_ITEMS_TOTAL.inc(status=status)
    BATCH_ITEM_DURATION.observe(duration_seconds, status=status)


def record_required_parameter_check(
    *,
    product_family: str,
    missing_fields: list[str] | tuple[str, ...],
) -> None:
    result = "missing" if missing_fields else "ok"
    REQUIRED_PARAMETER_CHECKS_TOTAL.inc(
        product_family=product_family,
        result=result,
    )
    for field_name in missing_fields:
        REQUIRED_PARAMETER_MISSING_TOTAL.inc(
            product_family=product_family,
            field=field_name,
        )


def render_metrics() -> str:
    lines: list[str] = []
    for metric in _METRICS:
        lines.extend(metric.render())
    return "\n".join(lines) + "\n"
