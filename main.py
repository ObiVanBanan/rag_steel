"""FastAPI application for the unified LD search API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_steel.config import DEFAULT_MODEL_NAME, RESULT_LIMIT_DEFAULT, RESULT_LIMIT_MAX
from rag_steel.search_engine import SearchEngine


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
    relevance_rating: float | None = None
    product: dict[str, Any] = Field(default_factory=dict)
    match_reasons: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SearchResponseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query: str
    count: int
    results: list[SearchResultResponse] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SearchEngine(model_name=DEFAULT_MODEL_NAME)
    try:
        yield
    finally:
        app.state.engine = None


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


def _effective_limit(request: LegacySearchRequest) -> int:
    return request.top_k or request.limit


def _project_result(result: Any) -> SearchResultResponse:
    return SearchResultResponse(
        rank=result.rank,
        relevance_rating=result.relevance_rating,
        product=result.product,
        match_reasons=list(result.match_reasons),
        mismatches=list(result.mismatches),
        source_evidence=list(result.source_evidence),
        score_breakdown=dict(result.score_breakdown),
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


@app.post("/v1/search", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
async def search_v1(
    request: SearchRequest,
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=request.limit)
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
    )


@app.post("/search", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
async def search_legacy(
    request: LegacySearchRequest,
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=_effective_limit(request))
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
    )


@app.post("/analogs", response_model=SearchResponseEnvelope, response_model_exclude_none=True)
async def find_analogs(
    request: LegacySearchRequest,
    engine: Annotated[SearchEngine, Depends(get_engine)],
) -> SearchResponseEnvelope:
    response = engine.search(request.query, limit=_effective_limit(request))
    return _build_response(
        query=response.query,
        include_debug=request.include_debug,
        engine_response=response,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
