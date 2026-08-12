"""Unified hybrid search over the active Qdrant collection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from time import perf_counter, sleep
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.embeddings import Embedder, create_embedder
from rag_steel.normalization import normalize_brand, normalize_connection, normalize_text
from rag_steel.query_constraints import QueryConstraints, extract_query_constraints
from rag_steel.runtime import (
    SearchBackendTimeoutError,
    SearchBackendUnavailableError,
)
from rag_steel.settings import (
    QDRANT_COLLECTION_ALIAS,
    QDRANT_URL,
    SOURCE_CANDIDATE_LIMIT,
    get_settings,
)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    id: str | int | None = None
    score: float | None = None
    product: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    source_evidence: list[dict[str, Any]] = Field(default_factory=list)


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    count: int
    results: list[SearchResult] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)


class CompetitorProduct(BaseModel):
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


class CompetitorMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_type: str
    differences: dict[str, Any] = Field(default_factory=dict)
    competitor: CompetitorProduct
    ld_articles: list[str] = Field(default_factory=list)


class SearchV2Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query: str
    status: str
    requested: dict[str, Any]
    results: list[CompetitorMatch] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)


@dataclass(slots=True)
class SearchEngine:
    embedder: Embedder | None = None
    qdrant_url: str = QDRANT_URL
    collection_alias: str = QDRANT_COLLECTION_ALIAS
    source_candidate_limit: int = SOURCE_CANDIDATE_LIMIT
    client: QdrantClient | None = None
    _client: QdrantClient | None = field(init=False, default=None, repr=False)
    _resolved_collection_name: str | None = field(init=False, default=None, repr=False)
    settings: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = self.client
        self._resolved_collection_name = None
        self.settings = get_settings()
        if self.embedder is None:
            self.embedder = create_embedder(self.settings)
        else:
            self.settings = replace(
                self.settings,
                embedding_model=self.embedder.model_name,
                embedding_dimension=self.embedder.dimension,
            )

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.qdrant_url,
                timeout=self.settings.qdrant_timeout_seconds,
            )
        return self._client

    @staticmethod
    def _is_missing_collection_error(exc: Exception, collection_name: str) -> bool:
        if not isinstance(exc, UnexpectedResponse):
            return False
        message = str(exc)
        return "404" in message and f"Collection `{collection_name}`" in message

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        message = str(exc).lower()
        timeout_markers = (
            "timeout",
            "timed out",
            "request timed out",
            "read timeout",
            "connect timeout",
        )
        return any(marker in message for marker in timeout_markers)

    @staticmethod
    def _is_retryable_qdrant_error(exc: Exception) -> bool:
        message = str(exc).lower()
        transient_markers = (
            "service unavailable",
            "temporarily unavailable",
            "failed to establish a new connection",
            "connection refused",
            "max retries exceeded",
            "timed out",
            "timeout",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return any(marker in message for marker in transient_markers)

    def _retry_delay_seconds(self, attempt: int) -> float:
        base_delay = max(0.0, self.settings.upstream_retry_base_delay_seconds)
        return base_delay * (2 ** max(0, attempt - 1))

    def _collection_name_candidates(self) -> list[str]:
        candidates = [self._resolved_collection_name, self.collection_alias]
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

    def _build_prefetches(self, query: str, dense_vector: list[float]) -> list[models.Prefetch]:
        return [
            models.Prefetch(
                query=dense_vector,
                using=self.settings.qdrant_dense_vector_name,
                limit=self.source_candidate_limit,
                score_threshold=self.settings.dense_score_threshold,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="qdrant/bm25",
                    options=models.Bm25Config(tokenizer=models.TokenizerType.MULTILINGUAL),
                ),
                using=self.settings.qdrant_sparse_vector_name,
                limit=self.source_candidate_limit,
                score_threshold=self.settings.bm25_score_threshold,
            ),
        ]

    def _query_points(self, client: QdrantClient, query: str, dense_vector: list[float]) -> Any:
        last_error: Exception | None = None
        for collection_name in self._collection_name_candidates():
            try:
                response = self._query_points_for_collection(
                    client,
                    collection_name,
                    query,
                    dense_vector,
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

    def _query_points_for_collection(
        self,
        client: QdrantClient,
        collection_name: str,
        query: str,
        dense_vector: list[float],
    ) -> Any:
        max_attempts = max(1, self.settings.upstream_max_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.query_points(
                    collection_name=collection_name,
                    prefetch=self._build_prefetches(query, dense_vector),
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.source_candidate_limit,
                    with_payload=True,
                )
                self._resolved_collection_name = collection_name
                return response
            except Exception as exc:
                if self._is_missing_collection_error(exc, collection_name):
                    raise
                if not self._is_retryable_qdrant_error(exc):
                    raise
                last_error = exc
                if attempt >= max_attempts:
                    break
                sleep(self._retry_delay_seconds(attempt))

        if last_error is not None:
            if self._is_timeout_error(last_error):
                raise SearchBackendTimeoutError("Qdrant query timed out") from last_error
            raise SearchBackendUnavailableError("Qdrant query failed") from last_error
        raise SearchBackendUnavailableError("Qdrant query failed")

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
        name = self._payload_text(payload, "name", "steel_name")
        return {
            "article": self._payload_text(payload, "article", "steel_article"),
            "article_norm": self._payload_text(payload, "article_norm", "steel_article_norm"),
            "name": name,
            "url": self._payload_text(payload, "url", "steel_url"),
            "price": payload.get("price") or payload.get("price_ld"),
            "dn": self._payload_number(payload, "dn", "steel_dn"),
            "pn_bar": self._payload_number(payload, "pn_bar", "steel_pn_bar"),
            "connection": self._payload_text(payload, "connection", "steel_connection"),
            "body_material": self._payload_text(payload, "body_material", "steel_body_material"),
            "medium": self._payload_text(payload, "medium", "steel_medium"),
            "control": self._payload_text(payload, "control", "steel_control"),
            "temperature": self._payload_text(payload, "temperature", "steel_temp"),
            "length_mm": self._payload_number(payload, "length_mm", "steel_length"),
            "brand": self._payload_text(payload, "brand", "steel_brand"),
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

    @staticmethod
    def _normalize_key_value(value: Any) -> str:
        return " ".join(str(value).split()).casefold().strip()

    def _source_product_matches_constraints(
        self,
        source_product: dict[str, Any],
        constraints: QueryConstraints,
    ) -> bool:
        source_pn = source_product.get("pn_bar")
        if constraints.pn_bar is not None and (
            source_pn is None or source_pn != float(constraints.pn_bar)
        ):
            return False
        if constraints.brand is not None:
            source_brand_value = source_product.get("brand") or source_product.get("name")
            if source_brand_value is None:
                return False
            source_brand = normalize_brand(source_brand_value)
            if source_brand is None or self._normalize_key_value(source_brand) != self._normalize_key_value(constraints.brand):
                return False
        if constraints.dn is not None:
            source_dn = source_product.get("dn")
            if source_dn is None or source_dn != float(constraints.dn):
                return False
        if constraints.connection is not None:
            source_connection_value = source_product.get("connection")
            if source_connection_value is None:
                return False
            source_connection = normalize_connection(source_connection_value)
            if source_connection is None or self._normalize_key_value(source_connection) != self._normalize_key_value(constraints.connection):
                return False
        if constraints.body_material is not None:
            source_body_material_value = source_product.get("body_material")
            if source_body_material_value is None:
                return False
            source_body_material = normalize_text(source_body_material_value)
            if source_body_material is None or self._normalize_key_value(source_body_material) != self._normalize_key_value(constraints.body_material):
                return False
        if constraints.series is not None:
            haystack = " ".join(
                value
                for value in [
                    source_product.get("name"),
                    source_product.get("article"),
                    source_product.get("article_norm"),
                ]
                if value
            )
            if not haystack:
                return False
            if not re.search(rf"\b{re.escape(constraints.series)}\b", normalize_text(haystack) or ""):
                return False
        return True

    def _filter_points_by_constraints(
        self,
        points: list[Any],
        constraints: QueryConstraints,
        *,
        pn_only: bool = False,
    ) -> list[Any]:
        filtered: list[Any] = []
        for point in points:
            payload = self._extract_payload(point)
            source_product = self._build_source_product(payload)
            if self._source_product_matches_constraints(source_product, constraints):
                filtered.append(point)
        return filtered

    @staticmethod
    def _source_product_key(source_product: dict[str, Any]) -> str:
        article_norm = source_product.get("article_norm")
        if article_norm:
            return str(article_norm)
        article = source_product.get("article")
        if article:
            return str(article)
        name = source_product.get("name")
        if name:
            return str(name)
        return ""

    def _build_competitor_product(self, source_product: dict[str, Any]) -> CompetitorProduct:
        return CompetitorProduct(
            article=source_product.get("article"),
            name=source_product.get("name"),
            brand=source_product.get("brand"),
            dn=source_product.get("dn"),
            pn_bar=source_product.get("pn_bar"),
            connection=source_product.get("connection"),
            medium=source_product.get("medium"),
            control=source_product.get("control"),
            body_material=source_product.get("body_material"),
            temperature=source_product.get("temperature"),
            length_mm=source_product.get("length_mm"),
            url=source_product.get("url"),
        )

    @staticmethod
    def _build_differences(
        constraints: QueryConstraints, source_product: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    def _collect_competitor_matches(
        self,
        points: list[Any],
        constraints: QueryConstraints,
        *,
        match_type: str,
    ) -> list[CompetitorMatch]:
        grouped: dict[str, dict[str, Any]] = {}

        for source_rank, point in enumerate(points, start=1):
            payload = self._extract_payload(point)
            source_score = self._extract_score(point)
            if source_score is None:
                continue
            source_product = self._build_source_product(payload)
            source_key = self._source_product_key(source_product)
            if not source_key:
                continue
            ld_articles = []
            for candidate in payload.get("ld_candidates") or []:
                if hasattr(candidate, "model_dump"):
                    candidate = candidate.model_dump(mode="json")
                product = self._build_ld_product(candidate)
                article = product.get("article")
                if article and article not in ld_articles:
                    ld_articles.append(article)

            current = grouped.get(source_key)
            if current is None:
                grouped[source_key] = {
                    "score": source_score,
                    "source_product": source_product,
                    "ld_articles": ld_articles,
                }
                continue

            current["score"] = max(float(current["score"]), source_score)
            current["ld_articles"] = list(dict.fromkeys([*current["ld_articles"], *ld_articles]))

        sorted_groups = sorted(
            grouped.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        results: list[CompetitorMatch] = []
        for group in sorted_groups:
            source_product = group["source_product"]
            results.append(
                CompetitorMatch(
                    match_type=match_type,
                    differences=self._build_differences(constraints, source_product),
                    competitor=self._build_competitor_product(source_product),
                    ld_articles=list(group["ld_articles"]),
                )
            )
        return results

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

    def _collect_ld_candidates(self, points: list[Any]) -> list[SearchResult]:
        deduplicated: dict[str, dict[str, Any]] = {}

        for source_rank, point in enumerate(points, start=1):
            payload = self._extract_payload(point)
            source_score = self._extract_score(point)
            if source_score is None:
                continue
            source_product = self._build_source_product(payload)
            source_evidence = self._build_source_evidence(
                source_article=source_product.get("article"),
                source_name=source_product.get("name"),
                source_score=source_score,
                source_rank=source_rank,
            )
            ld_candidates = payload.get("ld_candidates") or []
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
                    source_rank=source_evidence.get("source_rank", source_rank),
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

        sorted_candidates = sorted(
            deduplicated.values(),
            key=lambda item: item["score"] if item["score"] is not None else 0.0,
            reverse=True,
        )

        results: list[SearchResult] = []
        for rank, candidate in enumerate(sorted_candidates, start=1):
            score = candidate["score"]
            results.append(
                SearchResult(
                    rank=rank,
                    id=candidate["id"],
                    score=score,
                    product=candidate["product"],
                    source_evidence=candidate["source_evidence"],
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
        runtime_model = self.embedder.model_name
        runtime_revision = getattr(self.embedder, "embedding_revision", "")
        runtime_dimension = int(self.embedder.dimension)
        client = self._get_client()
        try:
            collection_name, collection_info = self._get_collection_info(client)
            metadata = self._extract_collection_metadata(collection_info)
            point_count = self._count_points(client, collection_name)
            dense_dimension = self._extract_dense_vector_dimension(collection_info)
        except Exception as exc:
            if not self._is_missing_collection_error(exc, self.collection_alias):
                raise
            details = {
                "runtime_model": runtime_model,
                "runtime_revision": runtime_revision,
                "runtime_dimension": runtime_dimension,
                "index_model": None,
                "index_revision": None,
                "index_dimension": None,
                "qdrant_dense_vector_dimension": None,
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
            }
            return False, {
                "status": "not_ready",
                "reason": "QDRANT_COLLECTION_MISSING",
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
                "details": details,
            }

        details = {
            "runtime_model": runtime_model,
            "runtime_revision": runtime_revision,
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
        if point_count <= 0:
            return False, {
                "status": "not_ready",
                "reason": "EMPTY_COLLECTION",
                "details": details,
            }
        if metadata.get("embedding_model") != runtime_model:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_INDEX_MISMATCH",
                "details": details,
            }
        if runtime_revision and metadata.get("embedding_revision") != runtime_revision:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_REVISION_MISMATCH",
                "details": details,
            }
        if metadata.get("embedding_dimension") != runtime_dimension:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_DIMENSION_MISMATCH",
                "details": details,
            }
        if dense_dimension != runtime_dimension:
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

    def _search_points(self, query: str) -> tuple[list[Any], list[Any], QueryConstraints, dict[str, float]]:
        timings: dict[str, float] = {}

        started = perf_counter()
        dense_vector = self.embedder.embed_query(query)
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._query_points(self._get_client(), query, dense_vector)
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        points = self._extract_points(response)
        constraints = extract_query_constraints(query)
        exact_points = self._filter_points_by_constraints(points, constraints)
        fallback_points = self._filter_points_by_constraints(points, constraints, pn_only=True)
        timings["total"] = sum(timings.values())
        return exact_points, fallback_points, constraints, timings

    def search(self, query: str, limit: int = 20, **_: Any) -> SearchResponse:
        timings: dict[str, float] = {}

        started = perf_counter()
        dense_vector = self.embedder.embed_query(query)
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._query_points(self._get_client(), query, dense_vector)
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        points = self._extract_points(response)
        results = self._collect_ld_candidates(points)
        results = results[: max(0, limit)]
        timings["ranking"] = (perf_counter() - started) * 1000.0
        timings["total"] = sum(v for key, v in timings.items() if key != "total")

        return SearchResponse(query=query, count=len(results), results=results, timing_ms=timings)

    def search_v2(self, query: str, limit: int = 20, **_: Any) -> SearchV2Response:
        timings: dict[str, float] = {}

        started = perf_counter()
        dense_vector = self.embedder.embed_query(query)
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._query_points(self._get_client(), query, dense_vector)
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        points = self._extract_points(response)
        constraints = extract_query_constraints(query)
        exact_points = self._filter_points_by_constraints(points, constraints)

        if exact_points:
            status = "exact_match"
            match_points = exact_points
        else:
            status = "not_found"
            match_points = []

        started = perf_counter()
        results = self._collect_competitor_matches(match_points, constraints, match_type=status)
        results = results[: max(0, limit)]
        timings["ranking"] = (perf_counter() - started) * 1000.0
        timings["total"] = sum(v for key, v in timings.items() if key != "total")

        return SearchV2Response(
            request_id=str(uuid4()),
            query=query,
            status=status,
            requested=constraints.model_dump(),
            results=results,
            timing_ms=timings,
        )


__all__ = [
    "CompetitorMatch",
    "CompetitorProduct",
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
    "SearchV2Response",
]
