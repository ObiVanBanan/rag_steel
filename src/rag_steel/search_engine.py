"""Unified hybrid search over the active Qdrant collection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.config import (
    DEFAULT_MODEL_NAME,
    MODEL_REGISTRY,
    QDRANT_COLLECTION_ALIAS,
    QDRANT_URL,
    SOURCE_CANDIDATE_LIMIT,
    get_embedding_model_spec,
    get_settings,
)
from rag_steel.embedding_text import EmbeddingTextAdapter


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    relevance_rating: float | None = None
    id: str | int | None = None
    score: float | None = None
    product: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)
    match_reasons: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
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
    _model: Any = field(init=False, default=None, repr=False)
    _client: QdrantClient | None = field(init=False, default=None, repr=False)
    _resolved_collection_name: str | None = field(init=False, default=None, repr=False)
    embedding_adapter: EmbeddingTextAdapter = field(init=False, repr=False)
    settings: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = None
        self._client = self.client
        self._resolved_collection_name = None
        self.settings = get_settings().for_model(self.model_name)
        self.embedding_adapter = EmbeddingTextAdapter(model_name=self.model_name)

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
    def _model_report_path(model_name: str) -> Path:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in model_name)
        normalized = "-".join(part for part in slug.split("-") if part)
        return Path("data/reports") / f"index_build_{normalized}.json"

    @staticmethod
    def _is_missing_collection_error(exc: Exception, collection_name: str) -> bool:
        if not isinstance(exc, UnexpectedResponse):
            return False
        message = str(exc)
        return "404" in message and f"Collection `{collection_name}`" in message

    def _fallback_collection_name_from_report(self) -> str | None:
        reports_dir = Path("data/reports")
        report_paths = [
            self._model_report_path(self.model_name),
            *sorted(reports_dir.glob("index_build*.json")),
        ]

        for report_path in report_paths:
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            report_alias = payload.get("collection_alias")
            report_collection = payload.get("collection_name")
            report_model = payload.get("embedding_model")
            if (
                report_alias != self.collection_alias
                or report_model != self.model_name
                or not isinstance(report_collection, str)
                or not report_collection
            ):
                continue
            return report_collection
        return None

    def _fallback_collection_metadata_from_report(self) -> dict[str, Any]:
        reports_dir = Path("data/reports")
        report_paths = [
            self._model_report_path(self.model_name),
            *sorted(reports_dir.glob("index_build*.json")),
        ]

        for report_path in report_paths:
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if payload.get("collection_alias") != self.collection_alias:
                continue
            if payload.get("embedding_model") != self.model_name:
                continue
            return payload
        return {}

    def _collection_name_candidates(self) -> list[str]:
        candidates = [self._resolved_collection_name, self.collection_alias]
        fallback = self._fallback_collection_name_from_report()
        if fallback:
            candidates.append(fallback)

        ordered: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _get_collection_info(self, client: QdrantClient) -> tuple[str, Any]:
        last_error: Exception | None = None
        for collection_name in self._collection_name_candidates():
            try:
                collection_info = client.get_collection(collection_name=collection_name)
            except Exception as exc:
                if self._is_missing_collection_error(exc, collection_name):
                    last_error = exc
                    continue
                raise
            self._resolved_collection_name = collection_name
            return collection_name, collection_info

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to resolve Qdrant collection")

    def _count_points(self, client: QdrantClient, collection_name: str) -> int:
        return int(client.count(collection_name=collection_name, exact=True).count)

    def _query_points(self, client: QdrantClient, query: str, dense_vector: list[float]) -> Any:
        last_error: Exception | None = None
        for collection_name in self._collection_name_candidates():
            try:
                response = client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using=self.settings.qdrant_dense_vector_name,
                            limit=self.source_candidate_limit,
                        ),
                        models.Prefetch(
                            query=self._sparse_query(query),
                            using=self.settings.qdrant_sparse_vector_name,
                            limit=self.source_candidate_limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.source_candidate_limit,
                    with_payload=True,
                )
            except Exception as exc:
                if self._is_missing_collection_error(exc, collection_name):
                    last_error = exc
                    continue
                raise
            self._resolved_collection_name = collection_name
            return response

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to resolve Qdrant collection")

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

    @staticmethod
    def _payload_text(payload: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _payload_number(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _product_key(product: dict[str, Any]) -> str | None:
        article_norm = product.get("article_norm")
        if article_norm:
            return str(article_norm)
        article = product.get("article")
        if article:
            return str(article)
        return None

    def _build_source_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "article": self._payload_text(payload, "article", "steel_article"),
            "article_norm": self._payload_text(payload, "article_norm", "steel_article_norm"),
            "name": self._payload_text(payload, "name", "steel_name"),
            "url": self._payload_text(payload, "url", "steel_url"),
            "price": payload.get("price") or payload.get("price_ld"),
            "dn": self._payload_number(payload, "dn", "steel_dn"),
            "pn_bar": self._payload_number(payload, "pn_bar", "steel_pn_bar"),
            "connection": self._payload_text(payload, "connection", "steel_connection"),
            "medium": self._payload_text(payload, "medium", "steel_medium"),
            "control": self._payload_text(payload, "control", "steel_control"),
        }

    def _build_ld_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "article": self._payload_text(payload, "article", "ld_article"),
            "article_norm": self._payload_text(payload, "article_norm", "ld_article_norm"),
            "name": self._payload_text(payload, "name", "ld_name"),
            "url": self._payload_text(payload, "url", "ld_url"),
            "price": payload.get("price") or payload.get("price_ld"),
            "dn": self._payload_number(payload, "dn", "ld_dn"),
            "pn_bar": self._payload_number(payload, "pn_bar", "ld_pn_mpa"),
            "connection": self._payload_text(payload, "connection", "ld_connection"),
            "medium": self._payload_text(payload, "medium", "ld_medium"),
            "control": self._payload_text(payload, "control", "ld_control"),
        }

    def _dense_query_text(self, query: str) -> str:
        return self.embedding_adapter.prepare_query(query)

    def _sparse_query(self, query: str) -> models.Document:
        return models.Document(
            text=query,
            model="qdrant/bm25",
            options=models.Bm25Config(
                tokenizer=models.TokenizerType.MULTILINGUAL,
            ),
        )

    def _encode_query(self, text: str) -> list[float]:
        model = self._get_model()
        encode_fn = getattr(model, "encode_query", None) or model.encode
        encode_kwargs = {
            "batch_size": 1,
            "normalize_embeddings": self.settings.embedding_normalize,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        try:
            vectors = encode_fn([text], **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("convert_to_numpy")
            vectors = encode_fn([text], **encode_kwargs)
        return self._coerce_vector(vectors)

    def _build_source_evidence(
        self,
        *,
        source_article: str | None,
        source_name: str | None,
        source_score: float | None,
        source_rank: int,
    ) -> dict[str, Any]:
        return {
            "source_article": source_article,
            "source_name": source_name,
            "source_score": source_score,
            "source_rank": source_rank,
        }

    def _rank_source_points(self, points: list[Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, point in enumerate(points, start=1):
            payload = self._extract_payload(point)
            source_score = self._extract_score(point)
            product = self._build_source_product(payload)
            evidence = self._build_source_evidence(
                source_article=product.get("article"),
                source_name=product.get("name"),
                source_score=source_score,
                source_rank=index,
            )
            results.append(
                SearchResult(
                    rank=index,
                    id=self._extract_id(point),
                    score=source_score,
                    product=product,
                    payload=payload,
                    source_evidence=[evidence],
                    score_breakdown={
                        "source_rrf_score": source_score if source_score is not None else 0.0
                    },
                )
            )

        results.sort(
            key=lambda item: (
                -(item.score if item.score is not None else float("-inf")),
                str(item.id),
            )
        )
        for rank, result in enumerate(results, start=1):
            result.rank = rank
            if result.source_evidence:
                result.source_evidence[0]["source_rank"] = rank
        return results

    def _collect_ld_candidates(self, source_results: list[SearchResult]) -> list[SearchResult]:
        deduplicated: dict[str, dict[str, Any]] = {}

        for source_result in source_results:
            source_score = source_result.score
            if source_score is None:
                continue
            source_evidence = source_result.source_evidence[0] if source_result.source_evidence else {}
            ld_candidates = source_result.payload.get("ld_candidates") or []
            for candidate in ld_candidates:
                if hasattr(candidate, "model_dump"):
                    candidate = candidate.model_dump(mode="json")
                product = self._build_ld_product(candidate)
                key = self._product_key(product)
                if key is None:
                    continue

                evidence = self._build_source_evidence(
                    source_article=source_evidence.get("source_article"),
                    source_name=source_evidence.get("source_name"),
                    source_score=source_score,
                    source_rank=source_evidence.get("source_rank", source_result.rank),
                )

                current = deduplicated.get(key)
                if current is None:
                    deduplicated[key] = {
                        "id": key,
                        "score": source_score,
                        "product": product,
                        "source_evidence": [evidence],
                    }
                    continue

                current["source_evidence"].append(evidence)
                if source_score > current["score"]:
                    current["score"] = source_score
                    current["product"] = product

        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                -(item["score"] if item["score"] is not None else float("-inf")),
                str(item["id"]),
            ),
        )

        results: list[SearchResult] = []
        for rank, candidate in enumerate(ordered, start=1):
            score = candidate["score"]
            results.append(
                SearchResult(
                    rank=rank,
                    id=candidate["id"],
                    score=score,
                    product=candidate["product"],
                    source_evidence=candidate["source_evidence"],
                    score_breakdown={
                        "source_rrf_score": score if score is not None else 0.0
                    },
                )
            )
        return results

    @staticmethod
    def _extract_collection_metadata(collection_info: Any) -> dict[str, Any]:
        if isinstance(collection_info, dict):
            return dict(collection_info.get("metadata") or {})
        metadata = getattr(collection_info, "metadata", None)
        return dict(metadata or {})

    def _extract_dense_vector_dimension(self, collection_info: Any) -> int | None:
        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if isinstance(collection_info, dict):
            vectors = ((collection_info.get("config") or {}).get("params") or {}).get("vectors")
        if vectors is None:
            return None
        if isinstance(vectors, dict):
            vector_config = vectors.get(self.settings.qdrant_dense_vector_name)
            if isinstance(vector_config, dict):
                size = vector_config.get("size")
            else:
                size = getattr(vector_config, "size", None)
            return int(size) if size is not None else None
        size = getattr(vectors, "size", None)
        return int(size) if size is not None else None

    def readiness_status(self) -> tuple[bool, dict[str, Any]]:
        runtime_spec = get_embedding_model_spec(self.model_name)
        model = self._get_model()
        runtime_dimension = int(model.get_sentence_embedding_dimension())
        client = self._get_client()
        collection_name, collection_info = self._get_collection_info(client)
        metadata = self._extract_collection_metadata(collection_info)
        report_metadata = self._fallback_collection_metadata_from_report()
        for key in ("embedding_model", "embedding_revision", "embedding_dimension"):
            if metadata.get(key) is None and report_metadata.get(key) is not None:
                metadata[key] = report_metadata[key]
        point_count = self._count_points(client, collection_name)
        dense_dimension = self._extract_dense_vector_dimension(collection_info)

        details = {
            "runtime_model": self.model_name,
            "runtime_revision": self.settings.embedding_revision,
            "runtime_dimension": runtime_dimension,
            "index_model": metadata.get("embedding_model"),
            "index_revision": metadata.get("embedding_revision"),
            "index_dimension": metadata.get("embedding_dimension"),
            "qdrant_dense_vector_dimension": dense_dimension,
            "collection_alias": self.collection_alias,
            "resolved_collection_name": collection_name,
            "point_count": point_count,
        }

        if runtime_dimension != self.settings.embedding_dimension:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_RUNTIME_DIMENSION_MISMATCH",
                "details": details,
            }
        if runtime_spec.dimension and runtime_dimension != runtime_spec.dimension:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_SPEC_DIMENSION_MISMATCH",
                "details": details,
            }
        if point_count <= 0:
            return False, {
                "status": "not_ready",
                "reason": "EMPTY_COLLECTION",
                "details": details,
            }
        if metadata.get("embedding_model") != self.model_name:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_INDEX_MISMATCH",
                "details": details,
            }
        if (
            self.settings.embedding_revision
            and metadata.get("embedding_revision") != self.settings.embedding_revision
        ):
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_REVISION_MISMATCH",
                "details": details,
            }
        if metadata.get("embedding_dimension") != self.settings.embedding_dimension:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_DIMENSION_MISMATCH",
                "details": details,
            }
        if dense_dimension != self.settings.embedding_dimension:
            return False, {
                "status": "not_ready",
                "reason": "QDRANT_VECTOR_DIMENSION_MISMATCH",
                "details": details,
            }

        return True, {
            "status": "ok",
            "collection_alias": self.collection_alias,
            "resolved_collection_name": collection_name,
            "point_count": point_count,
            "details": details,
        }

    def search(self, query: str, limit: int = 20, **_: Any) -> SearchResponse:
        timings: dict[str, float] = {}

        started = perf_counter()
        dense_query_text = self._dense_query_text(query)
        dense_vector = self._encode_query(dense_query_text)
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._query_points(self._get_client(), query, dense_vector)
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        points = self._extract_points(response)

        started = perf_counter()
        source_results = self._rank_source_points(points)
        results = self._collect_ld_candidates(source_results)
        results = results[: max(0, limit)]
        timings["ranking"] = (perf_counter() - started) * 1000.0

        timings["total"] = sum(timings.values())

        return SearchResponse(
            query=query,
            count=len(results),
            results=results,
            timing_ms=timings,
        )


__all__ = [
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
]
