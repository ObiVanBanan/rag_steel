"""FastAPI application for the unified LD search API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.observability import (
    PROMETHEUS_CONTENT_TYPE,
    dec_in_flight,
    dec_search_in_flight,
    get_request_id,
    inc_in_flight,
    inc_search_in_flight,
    log_batch_completed,
    log_batch_item_diagnostic,
    log_http_request_completed,
    record_api_error,
    record_batch_item,
    record_batch_size,
    record_http_request,
    record_required_parameter_check,
    render_metrics,
    reset_request_id,
    resolve_request_id,
    set_request_id,
)
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
    EmbeddingTimeoutError,
    EmbeddingUpstreamError,
    SearchBackendTimeoutError,
    SearchBackendUnavailableError,
    SearchBusyError,
    SearchConcurrencyGate,
)
from rag_steel.search_engine import SearchEngine
from rag_steel.search_messages import SEARCH_FAILURE_MESSAGE
from rag_steel.settings import RESULT_LIMIT_DEFAULT, RESULT_LIMIT_MAX, get_settings

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=RESULT_LIMIT_DEFAULT, ge=1, le=RESULT_LIMIT_MAX)
    include_debug: bool = False


class BatchSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    products: list[str] | None = Field(default=None, min_length=1, max_length=50)
    query: str | None = Field(default=None, min_length=1, max_length=512)
    limit: int = Field(default=RESULT_LIMIT_DEFAULT, ge=1, le=RESULT_LIMIT_MAX)

    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for index, product in enumerate(value):
            if not isinstance(product, str) or not product.strip():
                raise ValueError(f"products[{index}] must be a non-empty string")
            normalized.append(product.strip())
        return normalized

    @model_validator(mode="after")
    def validate_input_shape(self) -> "BatchSearchRequest":
        if (self.products is None) == (self.query is None):
            raise ValueError("Provide exactly one of products or query")
        return self


class LegacySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=RESULT_LIMIT_DEFAULT, ge=1, le=RESULT_LIMIT_MAX)
    top_k: int | None = Field(default=None, ge=1, le=RESULT_LIMIT_MAX)
    use_hybrid: bool = True
    include_debug: bool = False


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float | None = None
    product: dict[str, Any] = Field(default_factory=dict)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)


class SearchResponseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query: str
    count: int
    results: list[SearchResultResponse] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None


class V2CompetitorProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: str | None = None
    name: str | None = None
    brand: str | None = None
    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None
    medium: str | None = None
    control: str | None = None
    body_material: str | None = None
    temperature: str | None = None
    length_mm: float | None = None
    url: str | None = None


class V2CompetitorMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_type: str
    differences: dict[str, Any] = Field(default_factory=dict)
    competitor: V2CompetitorProduct
    ld_articles: list[str] = Field(default_factory=list)


class V2SearchResponseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query: str
    status: str
    requested: dict[str, Any] | None = None
    reason: dict[str, Any] | None = None
    results: list[V2CompetitorMatch] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)


class V2BatchSearchResponseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    count: int
    items: list[V2SearchResponseEnvelope] = Field(default_factory=list)
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.engine = SearchEngine()
    app.state.search_gate = SearchConcurrencyGate(settings.max_concurrent_searches)
    try:
        yield
    finally:
        app.state.engine = None
        app.state.search_gate = None


app = FastAPI(
    title="LD Analog Search API",
    description="Unified LD search over Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

SEARCH_ERROR_PATHS = {"/v1/search", "/search", "/analogs", "/v2/search"}


def _request_error_message(request: Request) -> str:
    if request.url.path in SEARCH_ERROR_PATHS:
        return SEARCH_FAILURE_MESSAGE
    return "Internal server error"


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = resolve_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    token = set_request_id(request_id)
    inc_in_flight()
    started = perf_counter()
    status_code = 500
    response: Response
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        logger.exception(
            "Unhandled exception while processing request",
            extra={"request_id": request_id},
        )
        record_api_error("INTERNAL_SERVER_ERROR")
        response = JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": _request_error_message(request),
                }
            },
        )
    finally:
        duration_ms = (perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        dec_in_flight()
        record_http_request(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_seconds=duration_ms / 1000.0,
        )
        log_http_request_completed(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        reset_request_id(token)
    return response


def get_engine(request: Request) -> SearchEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready")
    return engine


def get_search_gate(request: Request) -> SearchConcurrencyGate:
    gate = getattr(request.app.state, "search_gate", None)
    if gate is None:
        raise HTTPException(status_code=503, detail="Search gate is not ready")
    return gate


def acquire_search_slot(
    gate: Annotated[SearchConcurrencyGate, Depends(get_search_gate)],
):
    with gate.acquire():
        inc_search_in_flight()
        try:
            yield
        finally:
            dec_search_in_flight()


def _effective_limit(request: LegacySearchRequest) -> int:
    return request.top_k or request.limit


def _project_result(result: Any) -> SearchResultResponse:
    return SearchResultResponse(
        rank=result.rank,
        score=result.score,
        product=result.product,
        source_evidence=list(result.source_evidence),
    )


def _build_response(
    *,
    query: str,
    include_debug: bool,
    engine_response: Any,
) -> SearchResponseEnvelope:
    payload: dict[str, Any] = {
        "request_id": get_request_id() or uuid4().hex,
        "query": query,
        "count": engine_response.count,
        "results": [_project_result(result) for result in engine_response.results],
        "timing_ms": dict(engine_response.timing_ms),
    }
    if include_debug:
        payload["debug"] = {"pipeline": "raw_query_dense_bm25_rrf"}
    return SearchResponseEnvelope(**payload)


def _build_v2_response(*, engine_response: Any) -> V2SearchResponseEnvelope:
    payload: dict[str, Any] = {
        "request_id": get_request_id() or engine_response.request_id,
        "query": engine_response.query,
        "status": engine_response.status,
        "results": [
            V2CompetitorMatch(
                match_type=result.match_type,
                differences=result.differences,
                competitor=V2CompetitorProduct(**result.competitor.model_dump()),
                ld_articles=list(result.ld_articles),
            )
            for result in engine_response.results
        ],
        "timing_ms": dict(engine_response.timing_ms),
    }
    if getattr(engine_response, "requested", None) is not None:
        payload["requested"] = engine_response.requested
    if getattr(engine_response, "reason", None) is not None:
        payload["reason"] = engine_response.reason
    return V2SearchResponseEnvelope(**payload)


def _batch_error_code(exc: Exception) -> str:
    if isinstance(exc, DeepSeekTimeoutError):
        return "DEEPSEEK_TIMEOUT"
    if isinstance(exc, DeepSeekConfigurationError):
        return "DEEPSEEK_CONFIGURATION_MISSING"
    if isinstance(exc, DeepSeekInvalidResponseError):
        return "DEEPSEEK_INVALID_RESPONSE"
    if isinstance(exc, DeepSeekUpstreamError):
        return "DEEPSEEK_UNAVAILABLE"
    if isinstance(exc, EmbeddingTimeoutError):
        return "EMBEDDING_TIMEOUT"
    if isinstance(exc, EmbeddingUpstreamError):
        return "EMBEDDING_UNAVAILABLE"
    if isinstance(exc, SearchBackendTimeoutError):
        return "SEARCH_BACKEND_TIMEOUT"
    if isinstance(exc, (SearchBackendUnavailableError, UnexpectedResponse)):
        return "SEARCH_BACKEND_UNAVAILABLE"
    return "INTERNAL_SERVER_ERROR"


def _build_v2_technical_failure(query: str, exc: Exception) -> V2SearchResponseEnvelope:
    code = _batch_error_code(exc)
    record_api_error(code)
    return V2SearchResponseEnvelope(
        request_id=get_request_id() or uuid4().hex,
        query=query,
        status="technical_failure",
        reason={
            "code": code,
            "message": SEARCH_FAILURE_MESSAGE,
            "retryable": code
            in {
                "DEEPSEEK_TIMEOUT",
                "DEEPSEEK_UNAVAILABLE",
                "EMBEDDING_TIMEOUT",
                "EMBEDDING_UNAVAILABLE",
                "SEARCH_BACKEND_TIMEOUT",
                "SEARCH_BACKEND_UNAVAILABLE",
            },
        },
        results=[],
        timing_ms={},
    )


def _error_response(code: str, message: str, *, status_code: int) -> JSONResponse:
    record_api_error(code)
    headers = {"Retry-After": "1"} if status_code == 503 and code == "SERVICE_BUSY" else None
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


@app.exception_handler(SearchBusyError)
async def search_busy_handler(request: Request, __: SearchBusyError) -> JSONResponse:
    return _error_response("SERVICE_BUSY", _request_error_message(request), status_code=503)


@app.exception_handler(EmbeddingTimeoutError)
async def embedding_timeout_handler(request: Request, __: EmbeddingTimeoutError) -> JSONResponse:
    return _error_response("EMBEDDING_TIMEOUT", _request_error_message(request), status_code=504)


@app.exception_handler(EmbeddingUpstreamError)
async def embedding_upstream_handler(request: Request, __: EmbeddingUpstreamError) -> JSONResponse:
    return _error_response(
        "EMBEDDING_UNAVAILABLE", _request_error_message(request), status_code=503
    )


@app.exception_handler(DeepSeekTimeoutError)
async def deepseek_timeout_handler(request: Request, __: DeepSeekTimeoutError) -> JSONResponse:
    return _error_response("DEEPSEEK_TIMEOUT", _request_error_message(request), status_code=504)


@app.exception_handler(DeepSeekConfigurationError)
async def deepseek_configuration_handler(
    request: Request, __: DeepSeekConfigurationError
) -> JSONResponse:
    return _error_response(
        "DEEPSEEK_CONFIGURATION_MISSING",
        _request_error_message(request),
        status_code=503,
    )


@app.exception_handler(DeepSeekInvalidResponseError)
async def deepseek_invalid_response_handler(
    request: Request, __: DeepSeekInvalidResponseError
) -> JSONResponse:
    return _error_response(
        "DEEPSEEK_INVALID_RESPONSE",
        _request_error_message(request),
        status_code=502,
    )


@app.exception_handler(DeepSeekUpstreamError)
async def deepseek_upstream_handler(request: Request, __: DeepSeekUpstreamError) -> JSONResponse:
    return _error_response(
        "DEEPSEEK_UNAVAILABLE",
        _request_error_message(request),
        status_code=503,
    )


@app.exception_handler(SearchBackendTimeoutError)
async def search_timeout_handler(request: Request, __: SearchBackendTimeoutError) -> JSONResponse:
    return _error_response(
        "SEARCH_BACKEND_TIMEOUT", _request_error_message(request), status_code=504
    )


@app.exception_handler(SearchBackendUnavailableError)
async def search_backend_handler(
    request: Request, __: SearchBackendUnavailableError
) -> JSONResponse:
    return _error_response(
        "SEARCH_BACKEND_UNAVAILABLE", _request_error_message(request), status_code=503
    )


@app.exception_handler(UnexpectedResponse)
async def qdrant_response_handler(request: Request, __: UnexpectedResponse) -> JSONResponse:
    return _error_response(
        "SEARCH_BACKEND_UNAVAILABLE", _request_error_message(request), status_code=503
    )


@app.post("/v1/search", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
def search_v1(
    request: SearchRequest,
    _: Annotated[None, Depends(acquire_search_slot)],
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=request.limit)
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
    )


@app.post("/search", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
def search_legacy(
    request: LegacySearchRequest,
    _: Annotated[None, Depends(acquire_search_slot)],
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=_effective_limit(request))
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
    )


@app.post("/analogs", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
def find_analogs(
    request: LegacySearchRequest,
    _: Annotated[None, Depends(acquire_search_slot)],
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=_effective_limit(request))
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
    )


@app.post(
    "/v2/search",
    response_model=V2BatchSearchResponseEnvelope | V2SearchResponseEnvelope,
    response_model_exclude_none=True,
)
def search_v2(
    request: BatchSearchRequest,
    _: Annotated[None, Depends(acquire_search_slot)],
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> V2BatchSearchResponseEnvelope | V2SearchResponseEnvelope:
    if request.query is not None:
        response = engine.search_v2(request.query, limit=request.limit)
        return _build_v2_response(engine_response=response)

    products = request.products or []
    batch_started = perf_counter()
    record_batch_size(len(products))
    items: list[V2SearchResponseEnvelope] = []
    status_counts: dict[str, int] = {}

    for item_index, product in enumerate(products):
        item_started = perf_counter()
        try:
            item = _build_v2_response(
                engine_response=engine.search_v2(product, limit=request.limit)
            )
        except Exception as exc:
            logger.exception(
                "V2 batch item failed",
                extra={"item_index": item_index, "query": product},
            )
            item = _build_v2_technical_failure(product, exc)

        item_duration_seconds = perf_counter() - item_started
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
        record_batch_item(item.status, item_duration_seconds)

        product_family = None
        missing_fields: list[str] = []
        error_code = None
        if item.requested:
            product_family = item.requested.get("product_family")
        if item.reason:
            product_family = item.reason.get("product_family") or product_family
            missing_fields = list(item.reason.get("missing_fields") or [])
            error_code = item.reason.get("code")

        if product_family:
            record_required_parameter_check(
                product_family=str(product_family),
                missing_fields=missing_fields,
            )

        log_batch_item_diagnostic(
            item_index=item_index,
            input_query=product,
            status=item.status,
            duration_ms=item_duration_seconds * 1000.0,
            product_family=str(product_family) if product_family else None,
            missing_fields=missing_fields,
            results_count=len(item.results),
            error_code=str(error_code) if error_code else None,
        )
        items.append(item)

    latency_ms = round((perf_counter() - batch_started) * 1000.0, 3)
    log_batch_completed(
        batch_size=len(products),
        duration_ms=latency_ms,
        status_counts=status_counts,
    )
    return V2BatchSearchResponseEnvelope(
        request_id=get_request_id() or uuid4().hex,
        count=len(items),
        items=items,
        latency_ms=latency_ms,
    )


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready(
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> dict[str, Any]:
    try:
        ready, payload = engine.readiness_status()
    except Exception as exc:  # pragma: no cover - exercised by integration tests
        raise HTTPException(status_code=503, detail="Search backend is not ready") from exc

    if ready:
        return payload
    return JSONResponse(status_code=503, content=payload)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type=PROMETHEUS_CONTENT_TYPE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
