"""Unified hybrid search over the active Qdrant collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from config import (
    DEFAULT_MODEL_NAME,
    MODEL_REGISTRY,
    QDRANT_COLLECTION_ALIAS,
    QDRANT_URL,
    SOURCE_CANDIDATE_LIMIT,
)
from rag_steel.query_processor import EmbeddingTextAdapter, ProcessedQuery, QueryProcessor


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    id: str | int | None = None
    score: float | None = None
    hybrid_score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)
    match_reasons: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    processed_query: ProcessedQuery
    count: int
    results: list[SearchResult] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)


@dataclass(slots=True)
class SearchEngine:
    model_name: str = DEFAULT_MODEL_NAME
    qdrant_url: str = QDRANT_URL
    collection_alias: str = QDRANT_COLLECTION_ALIAS
    source_candidate_limit: int = SOURCE_CANDIDATE_LIMIT
    model_factory: Callable[[], Any] | None = None
    client: QdrantClient | None = None
    query_processor: QueryProcessor | None = None
    _model: Any = field(init=False, default=None, repr=False)
    _client: QdrantClient | None = field(init=False, default=None, repr=False)
    embedding_adapter: EmbeddingTextAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = None
        self._client = self.client
        self.query_processor = self.query_processor or QueryProcessor(
            model_name=self.model_name
        )
        self.embedding_adapter = self.query_processor.embedding_adapter

    def _get_model(self) -> Any:
        if self._model is None:
            factory = self.model_factory or MODEL_REGISTRY[self.model_name]
            self._model = factory()
        return self._model

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    @staticmethod
    def _coerce_vector(vectors: Any) -> list[float]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if vectors and isinstance(vectors[0], list):
            return list(vectors[0])
        return list(vectors)

    @staticmethod
    def _extract_points(response: Any) -> list[Any]:
        points = getattr(response, "points", None)
        if points is None:
            points = getattr(response, "result", None)
        if points is None:
            return []
        return list(points)

    @staticmethod
    def _extract_payload(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if payload is None and isinstance(point, dict):
            payload = point.get("payload")
        return dict(payload or {})

    @staticmethod
    def _extract_id(point: Any) -> str | int | None:
        if isinstance(point, dict):
            return point.get("id")
        return getattr(point, "id", None)

    @staticmethod
    def _extract_score(point: Any) -> float | None:
        score = point.get("score") if isinstance(point, dict) else getattr(point, "score", None)
        return float(score) if score is not None else None

    def _dense_query_text(self, processed: ProcessedQuery) -> str:
        return self.embedding_adapter.prepare_query(processed.semantic_text)

    def _sparse_query(self, processed: ProcessedQuery) -> models.Document:
        return models.Document(
            text=processed.lexical_text,
            model="qdrant/bm25",
            options=models.Bm25Config(
                tokenizer=models.TokenizerType.MULTILINGUAL,
            ),
        )

    def search(self, query: str, limit: int = 20, **_: Any) -> SearchResponse:
        timings: dict[str, float] = {}

        started = perf_counter()
        processed = self.query_processor.process(query)
        timings["normalize"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        dense_query_text = self._dense_query_text(processed)
        dense_vector = self._coerce_vector(
            self._get_model().encode(
                [dense_query_text],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._get_client().query_points(
            collection_name=self.collection_alias,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=self.source_candidate_limit,
                ),
                models.Prefetch(
                    query=self._sparse_query(processed),
                    using="sparse",
                    limit=self.source_candidate_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.source_candidate_limit,
            with_payload=True,
        )
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        points = self._extract_points(response)[: max(0, limit)]
        results: list[SearchResult] = []
        for index, point in enumerate(points, start=1):
            score = self._extract_score(point)
            payload = self._extract_payload(point)
            results.append(
                SearchResult(
                    rank=index,
                    id=self._extract_id(point),
                    score=score,
                    hybrid_score=score,
                    payload=payload,
                    score_breakdown={"hybrid_score": score} if score is not None else {},
                )
            )

        timings["total"] = sum(timings.values())

        return SearchResponse(
            query=query,
            processed_query=processed,
            count=len(results),
            results=results,
            timing_ms=timings,
        )


__all__ = [
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
]
