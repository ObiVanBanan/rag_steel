"""FastAPI application for the unified LD search API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.runtime import (
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
from rag_steel.settings import RESULT_LIMIT_DEFAULT, RESULT_LIMIT_MAX, get_settings


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=RESULT_LIMIT_DEFAULT, ge=1, le=RESULT_LIMIT_MAX)
    include_debug: bool = False


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
        yield


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
        "request_id": str(uuid4()),
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
        "request_id": engine_response.request_id,
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


def _error_response(code: str, message: str, *, status_code: int) -> JSONResponse:
    headers = {"Retry-After": "1"} if status_code == 503 and code == "SERVICE_BUSY" else None
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


@app.exception_handler(SearchBusyError)
async def search_busy_handler(_: Request, __: SearchBusyError) -> JSONResponse:
    return _error_response("SERVICE_BUSY", "Search service is temporarily busy", status_code=503)


@app.exception_handler(EmbeddingTimeoutError)
async def embedding_timeout_handler(_: Request, __: EmbeddingTimeoutError) -> JSONResponse:
    return _error_response("EMBEDDING_TIMEOUT", "Embedding upstream timed out", status_code=504)


@app.exception_handler(EmbeddingUpstreamError)
async def embedding_upstream_handler(_: Request, __: EmbeddingUpstreamError) -> JSONResponse:
    return _error_response(
        "EMBEDDING_UNAVAILABLE", "Embedding upstream is unavailable", status_code=503
    )


@app.exception_handler(DeepSeekTimeoutError)
async def deepseek_timeout_handler(_: Request, __: DeepSeekTimeoutError) -> JSONResponse:
    return _error_response("DEEPSEEK_TIMEOUT", "DeepSeek request timed out", status_code=504)


@app.exception_handler(DeepSeekUpstreamError)
async def deepseek_upstream_handler(_: Request, __: DeepSeekUpstreamError) -> JSONResponse:
    return _error_response(
        "DEEPSEEK_UNAVAILABLE",
        "DeepSeek upstream is unavailable",
        status_code=503,
    )


@app.exception_handler(SearchBackendTimeoutError)
async def search_timeout_handler(_: Request, __: SearchBackendTimeoutError) -> JSONResponse:
    return _error_response("SEARCH_BACKEND_TIMEOUT", "Search backend timed out", status_code=504)


@app.exception_handler(SearchBackendUnavailableError)
async def search_backend_handler(_: Request, __: SearchBackendUnavailableError) -> JSONResponse:
    return _error_response(
        "SEARCH_BACKEND_UNAVAILABLE", "Search backend is unavailable", status_code=503
    )


@app.exception_handler(UnexpectedResponse)
async def qdrant_response_handler(_: Request, __: UnexpectedResponse) -> JSONResponse:
    return _error_response(
        "SEARCH_BACKEND_UNAVAILABLE", "Search backend is unavailable", status_code=503
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


@app.post("/v2/search", response_model=V2SearchResponseEnvelope, response_model_exclude_none=True)
def search_v2(
    request: SearchRequest,
    _: Annotated[None, Depends(acquire_search_slot)],
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> V2SearchResponseEnvelope:
    response = engine.search_v2(request.query, limit=request.limit)
    return _build_v2_response(engine_response=response)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
