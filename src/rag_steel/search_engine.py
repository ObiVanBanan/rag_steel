"""Unified hybrid search over the active Qdrant collection."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter, sleep
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.attribute_extractor import ExtractedAttributes, create_attribute_extractor
from rag_steel.brand_gate import detect_competitor_brand
from rag_steel.embeddings import Embedder, create_embedder
from rag_steel.index_metadata import check_index_compatibility
from rag_steel.normalization import (
    normalize_body_material,
    normalize_brand,
    normalize_connection,
    normalize_text,
)
from rag_steel.observability import (
    get_request_id,
    log_search_completed,
    record_deepseek_error,
    record_deepseek_request,
    record_embedding_error,
    record_embedding_request,
    record_qdrant_error,
    record_qdrant_request,
    record_ranking_duration,
    record_search_request,
)
from rag_steel.query_constraints import QueryConstraints
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
    EmbeddingTimeoutError,
    EmbeddingUpstreamError,
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
    requested: dict[str, Any] | None = None
    reason: dict[str, Any] | None = None
    results: list[CompetitorMatch] = Field(default_factory=list)
    timing_ms: dict[str, float] = Field(default_factory=dict)


def _deepseek_error_type(exc: Exception) -> str:
    if isinstance(exc, DeepSeekTimeoutError):
        return "timeout"
    if isinstance(exc, DeepSeekUpstreamError):
        return "upstream"
    if isinstance(exc, DeepSeekInvalidResponseError):
        return "invalid_response"
    if isinstance(exc, DeepSeekConfigurationError):
        return "configuration"
    return "unexpected"


def _embedding_error_type(exc: Exception) -> str:
    if isinstance(exc, EmbeddingTimeoutError):
        return "timeout"
    if isinstance(exc, EmbeddingUpstreamError):
        return "upstream"
    return "unexpected"


def _qdrant_error_type(exc: Exception) -> str:
    if isinstance(exc, SearchBackendTimeoutError):
        return "timeout"
    if isinstance(exc, SearchBackendUnavailableError):
        return "upstream"
    return "unexpected"


@dataclass(slots=True)
class SearchEngine:
    embedder: Embedder | None = None
    brand_detector: Callable[[str], str | None] | None = None
    attribute_extractor: Any | None = None
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
        if self.brand_detector is None:
            self.brand_detector = detect_competitor_brand
        if self.attribute_extractor is None:
            self.attribute_extractor = create_attribute_extractor(self.settings)
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
        candidates = [self.collection_alias, self._resolved_collection_name]
        ordered: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _resolve_physical_collection_name(self, client: QdrantClient, collection_name: str) -> str:
        if collection_name != self.collection_alias:
            return collection_name
        try:
            aliases = client.get_aliases()
        except Exception:
            return collection_name
        for alias in getattr(aliases, "aliases", []):
            if getattr(alias, "alias_name", None) == collection_name:
                resolved_collection_name = getattr(alias, "collection_name", None)
                if resolved_collection_name:
                    return str(resolved_collection_name)
        return collection_name

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
            resolved_collection_name = self._resolve_physical_collection_name(
                client,
                collection_name,
            )
            self._resolved_collection_name = resolved_collection_name
            return resolved_collection_name, collection_info

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

    def _query_points(
        self,
        client: QdrantClient,
        query: str,
        dense_vector: list[float],
        *,
        query_filter: models.Filter | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for collection_name in self._collection_name_candidates():
            try:
                response = self._query_points_for_collection(
                    client,
                    collection_name,
                    query,
                    dense_vector,
                    query_filter=query_filter,
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
        *,
        query_filter: models.Filter | None = None,
    ) -> Any:
        max_attempts = max(1, self.settings.upstream_max_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.query_points(
                    collection_name=collection_name,
                    prefetch=self._build_prefetches(query, dense_vector),
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=query_filter,
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
    def _build_cannot_process_response(query: str) -> SearchV2Response:
        return SearchV2Response(
            request_id=get_request_id() or str(uuid4()),
            query=query,
            status="cannot_process",
            requested={"brand": None},
            reason={
                "code": "COMPETITOR_BRAND_REQUIRED",
                "message": "В запросе не указана поддерживаемая торговая марка конкурента.",
                "retryable": False,
            },
            results=[],
            timing_ms={},
        )

    @staticmethod
    def _build_not_found_response(
        *,
        query: str,
        requested: dict[str, Any],
        timing_ms: dict[str, float],
    ) -> SearchV2Response:
        return SearchV2Response(
            request_id=get_request_id() or str(uuid4()),
            query=query,
            status="not_found",
            requested=requested,
            reason={
                "code": "NOT_FOUND",
                "message": "Подходящие товары не найдены.",
                "retryable": False,
            },
            results=[],
            timing_ms=timing_ms,
        )

    @staticmethod
    def _attributes_to_requested(
        *,
        brand: str,
        attributes: ExtractedAttributes,
    ) -> dict[str, Any]:
        requested = attributes.model_dump()
        requested["brand"] = brand
        return requested

    @staticmethod
    def _hard_constraints_from_attributes(
        *,
        brand: str,
        attributes: ExtractedAttributes,
    ) -> QueryConstraints:
        return QueryConstraints(
            brand=brand,
            dn=int(attributes.dn) if attributes.dn is not None else None,
            pn_bar=int(attributes.pn_bar) if attributes.pn_bar is not None else None,
            connection=attributes.connection,
            series=None,
            body_material=None,
        )

    @staticmethod
    def _build_query_filter(constraints: QueryConstraints) -> models.Filter | None:
        must: list[Any] = []
        if constraints.brand is not None:
            must.append(
                models.FieldCondition(
                    key="brand",
                    match=models.MatchValue(value=constraints.brand),
                )
            )
        if constraints.dn is not None:
            must.append(
                models.FieldCondition(
                    key="dn",
                    range=models.Range(gte=float(constraints.dn), lte=float(constraints.dn)),
                )
            )
        if constraints.pn_bar is not None:
            must.append(
                models.FieldCondition(
                    key="pn_bar",
                    range=models.Range(
                        gte=float(constraints.pn_bar),
                        lte=float(constraints.pn_bar),
                    ),
                )
            )
        if constraints.connection is not None:
            must.append(
                models.FieldCondition(
                    key="connection",
                    match=models.MatchValue(value=constraints.connection),
                )
            )
        if not must:
            return None
        return models.Filter(must=must)

    @staticmethod
    def _normalize_soft_attribute(value: Any) -> str | None:
        if value is None:
            return None
        text = normalize_text(value)
        return text

    def _build_retrieval_query(
        self,
        query: str,
        *,
        brand: str,
        attributes: ExtractedAttributes,
    ) -> str:
        parts = [query.strip(), brand]
        for value in (
            attributes.body_material,
            attributes.medium,
            attributes.control,
            attributes.temperature,
            attributes.series,
            attributes.article,
        ):
            normalized = self._normalize_soft_attribute(value)
            if normalized:
                parts.append(normalized)
        if attributes.length_mm is not None:
            parts.append(f"{attributes.length_mm:g} мм")
        return " ".join(part for part in parts if part)

    def _extract_attributes(self, query: str) -> ExtractedAttributes:
        if self.attribute_extractor is not None:
            extracted = self.attribute_extractor.extract(query)
            if isinstance(extracted, ExtractedAttributes):
                return extracted
            if hasattr(extracted, "model_dump"):
                return ExtractedAttributes.model_validate(extracted.model_dump())
            return ExtractedAttributes.model_validate(extracted)
        raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is required for V2 attribute extraction")

    @staticmethod
    def _validate_hard_constraints(
        source_product: dict[str, Any],
        constraints: QueryConstraints,
    ) -> bool:
        if constraints.brand is not None:
            source_brand_value = source_product.get("brand") or source_product.get("name")
            if source_brand_value is None:
                return False
            source_brand = normalize_brand(source_brand_value)
            if source_brand is None or SearchEngine._normalize_key_value(
                source_brand
            ) != SearchEngine._normalize_key_value(constraints.brand):
                return False
        if constraints.dn is not None:
            source_dn = source_product.get("dn")
            if source_dn is None or source_dn != float(constraints.dn):
                return False
        if constraints.pn_bar is not None:
            source_pn = source_product.get("pn_bar")
            if source_pn is None or source_pn != float(constraints.pn_bar):
                return False
        if constraints.connection is not None:
            source_connection_value = source_product.get("connection")
            if source_connection_value is None:
                return False
            source_connection = normalize_connection(source_connection_value)
            if source_connection is None or SearchEngine._normalize_key_value(
                source_connection
            ) != SearchEngine._normalize_key_value(constraints.connection):
                return False
        return True

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
            if source_brand is None or self._normalize_key_value(
                source_brand
            ) != self._normalize_key_value(constraints.brand):
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
            if source_connection is None or self._normalize_key_value(
                source_connection
            ) != self._normalize_key_value(constraints.connection):
                return False
        if constraints.body_material is not None:
            source_body_material_value = source_product.get("body_material")
            if source_body_material_value is None:
                return False
            source_body_material = normalize_body_material(source_body_material_value)
            if source_body_material is None or self._normalize_key_value(
                source_body_material
            ) != self._normalize_key_value(constraints.body_material):
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
            if not re.search(
                rf"\b{re.escape(constraints.series)}\b", normalize_text(haystack) or ""
            ):
                return False
        return True

    def _filter_points_by_constraints(
        self,
        points: list[Any],
        constraints: QueryConstraints,
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

        for _source_rank, point in enumerate(points, start=1):
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
            metadata = collection_info.get("metadata")
            if metadata:
                return dict(metadata)
            config = collection_info.get("config") or {}
            config_metadata = config.get("metadata") or {}
            if config_metadata:
                return dict(config_metadata)
            params = config.get("params") or {}
            return dict(params.get("metadata") or {})
        metadata = getattr(collection_info, "metadata", None)
        if metadata:
            return dict(metadata)
        config = getattr(collection_info, "config", None)
        config_metadata = getattr(config, "metadata", None)
        if config_metadata:
            return dict(config_metadata)
        params = getattr(config, "params", None)
        metadata = getattr(params, "metadata", None)
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
        deepseek_configured = bool(self.settings.deepseek_api_key)
        if not deepseek_configured:
            qdrant = {
                "alias": self.collection_alias,
                "resolved_collection": None,
                "point_count": 0,
                "vector_dimension": None,
            }
            index = {
                "compatible": False,
                "reason": "DEEPSEEK_CONFIGURATION_MISSING",
                "schema_version": None,
                "index_format_version": None,
                "embedding_model": None,
                "embedding_dimension": None,
                "warnings": [],
            }
            details = {
                "runtime_model": runtime_model,
                "runtime_revision": runtime_revision,
                "runtime_dimension": runtime_dimension,
                "index_schema_version": None,
                "index_model": None,
                "index_revision": None,
                "index_dimension": None,
                "qdrant_dense_vector_dimension": None,
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
                "deepseek_configured": False,
                "deepseek_model": self.settings.deepseek_model,
            }
            return False, {
                "status": "not_ready",
                "reason": "DEEPSEEK_CONFIGURATION_MISSING",
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
                "qdrant": qdrant,
                "index": index,
                "details": details,
            }
        client = self._get_client()
        try:
            collection_name, collection_info = self._get_collection_info(client)
            metadata = self._extract_collection_metadata(collection_info)
            point_count = self._count_points(client, collection_name)
            dense_dimension = self._extract_dense_vector_dimension(collection_info)
        except Exception as exc:
            if not self._is_missing_collection_error(exc, self.collection_alias):
                raise
            qdrant = {
                "alias": self.collection_alias,
                "resolved_collection": None,
                "point_count": 0,
                "vector_dimension": None,
            }
            index = {
                "compatible": False,
                "reason": "QDRANT_COLLECTION_MISSING",
                "schema_version": None,
                "index_format_version": None,
                "embedding_model": None,
                "embedding_dimension": None,
                "warnings": [],
            }
            details = {
                "runtime_model": runtime_model,
                "runtime_revision": runtime_revision,
                "runtime_dimension": runtime_dimension,
                "index_schema_version": None,
                "index_model": None,
                "index_revision": None,
                "index_dimension": None,
                "qdrant_dense_vector_dimension": None,
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
                "deepseek_configured": deepseek_configured,
                "deepseek_model": self.settings.deepseek_model,
            }
            return False, {
                "status": "not_ready",
                "reason": "QDRANT_COLLECTION_MISSING",
                "collection_alias": self.collection_alias,
                "resolved_collection_name": None,
                "point_count": 0,
                "qdrant": qdrant,
                "index": index,
                "details": details,
            }

        compatibility = check_index_compatibility(
            metadata=metadata,
            actual_dimension=dense_dimension,
            settings=self.settings,
        )
        qdrant = {
            "alias": self.collection_alias,
            "resolved_collection": collection_name,
            "point_count": point_count,
            "vector_dimension": dense_dimension,
        }
        index = {
            "compatible": compatibility["compatible"],
            "reason": compatibility["reason"],
            "schema_version": compatibility["schema_version"],
            "index_format_version": compatibility["index_format_version"],
            "embedding_model": compatibility["embedding_model"],
            "embedding_dimension": compatibility["embedding_dimension"],
            "warnings": list(compatibility["warnings"]),
        }
        details = {
            "runtime_model": runtime_model,
            "runtime_revision": runtime_revision,
            "runtime_dimension": runtime_dimension,
            "index_schema_version": metadata.get("index_schema_version"),
            "index_model": metadata.get("embedding_model"),
            "index_revision": metadata.get("embedding_revision"),
            "index_dimension": metadata.get("embedding_dimension"),
            "qdrant_dense_vector_dimension": dense_dimension,
            "collection_alias": self.collection_alias,
            "resolved_collection_name": collection_name,
            "point_count": point_count,
            "deepseek_configured": deepseek_configured,
            "deepseek_model": self.settings.deepseek_model,
        }

        if runtime_dimension != self.settings.embedding_dimension:
            return False, {
                "status": "not_ready",
                "reason": "EMBEDDING_RUNTIME_DIMENSION_MISMATCH",
                "details": details,
                "qdrant": qdrant,
                "index": index,
            }
        if point_count <= 0:
            return False, {
                "status": "not_ready",
                "reason": "EMPTY_COLLECTION",
                "details": details,
                "qdrant": qdrant,
                "index": index,
            }
        if not compatibility["compatible"]:
            return False, {
                "status": "not_ready",
                "reason": compatibility["reason"],
                "details": details,
                "qdrant": qdrant,
                "index": index,
            }

        return True, {
            "status": "ok",
            "collection_alias": self.collection_alias,
            "resolved_collection_name": collection_name,
            "point_count": point_count,
            "qdrant": qdrant,
            "index": index,
            "details": details,
        }

    def search(self, query: str, limit: int = 20, **_: Any) -> SearchResponse:
        timings: dict[str, float] = {}

        started = perf_counter()
        try:
            dense_vector = self.embedder.embed_query(query)
        except Exception as exc:
            elapsed = perf_counter() - started
            record_embedding_error(_embedding_error_type(exc), elapsed)
            raise
        timings["embedding"] = (perf_counter() - started) * 1000.0
        record_embedding_request(timings["embedding"] / 1000.0)

        started = perf_counter()
        try:
            response = self._query_points(self._get_client(), query, dense_vector)
        except Exception as exc:
            elapsed = perf_counter() - started
            record_qdrant_error(_qdrant_error_type(exc), elapsed)
            raise
        timings["qdrant"] = (perf_counter() - started) * 1000.0
        record_qdrant_request(timings["qdrant"] / 1000.0)

        started = perf_counter()
        try:
            points = self._extract_points(response)
            results = self._collect_ld_candidates(points)
            results = results[: max(0, limit)]
        finally:
            timings["ranking"] = (perf_counter() - started) * 1000.0
            record_ranking_duration(timings["ranking"] / 1000.0)
        timings["total"] = sum(v for key, v in timings.items() if key != "total")

        return SearchResponse(query=query, count=len(results), results=results, timing_ms=timings)

    def search_v2(self, query: str, limit: int = 20, **_: Any) -> SearchV2Response:
        timings: dict[str, float] = {}
        total_started = perf_counter()
        result_status = "technical_failure"
        results: list[CompetitorMatch] = []

        def _finalize(*, status: str, requested: dict[str, Any] | None, results_count: int) -> None:
            record_search_request(status)
            log_search_completed(
                request_id=get_request_id(),
                result_status=status,
                results_count=results_count,
                total_ms=(perf_counter() - total_started) * 1000.0,
                deepseek_ms=timings.get("deepseek"),
                embedding_ms=timings.get("embedding"),
                qdrant_ms=timings.get("qdrant"),
                ranking_ms=timings.get("ranking"),
                query_length=len(query),
            )

        try:
            brand = self.brand_detector(query) if self.brand_detector is not None else None
            if brand is None:
                result_status = "cannot_process"
                _finalize(status=result_status, requested={"brand": None}, results_count=0)
                return self._build_cannot_process_response(query)

            deepseek_started = perf_counter()
            try:
                attributes = self._extract_attributes(query)
            except Exception as exc:
                timings["deepseek"] = (perf_counter() - deepseek_started) * 1000.0
                record_deepseek_error(_deepseek_error_type(exc), timings["deepseek"] / 1000.0)
                raise
            timings["deepseek"] = (perf_counter() - deepseek_started) * 1000.0
            record_deepseek_request(timings["deepseek"] / 1000.0)

            hard_constraints = self._hard_constraints_from_attributes(
                brand=brand,
                attributes=attributes,
            )
            requested = self._attributes_to_requested(brand=brand, attributes=attributes)
            query_filter = self._build_query_filter(hard_constraints)
            retrieval_query = self._build_retrieval_query(
                query,
                brand=brand,
                attributes=attributes,
            )

            started = perf_counter()
            try:
                dense_vector = self.embedder.embed_query(retrieval_query)
            except Exception as exc:
                timings["embedding"] = (perf_counter() - started) * 1000.0
                record_embedding_error(_embedding_error_type(exc), timings["embedding"] / 1000.0)
                raise
            timings["embedding"] = (perf_counter() - started) * 1000.0
            record_embedding_request(timings["embedding"] / 1000.0)

            started = perf_counter()
            try:
                response = self._query_points(
                    self._get_client(),
                    retrieval_query,
                    dense_vector,
                    query_filter=query_filter,
                )
            except Exception as exc:
                timings["qdrant"] = (perf_counter() - started) * 1000.0
                record_qdrant_error(_qdrant_error_type(exc), timings["qdrant"] / 1000.0)
                raise
            timings["qdrant"] = (perf_counter() - started) * 1000.0
            record_qdrant_request(timings["qdrant"] / 1000.0)

            points = self._extract_points(response)
            filtered_points = self._filter_points_by_constraints(points, hard_constraints)
            if not filtered_points:
                timings["ranking"] = 0.0
                result_status = "not_found"
                _finalize(status=result_status, requested=requested, results_count=0)
                timings["total"] = sum(v for key, v in timings.items() if key != "total")
                return self._build_not_found_response(
                    query=query,
                    requested=requested,
                    timing_ms=timings,
                )

            started = perf_counter()
            try:
                results = self._collect_competitor_matches(
                    filtered_points,
                    hard_constraints,
                    match_type="exact_match",
                )
                results = results[: max(0, limit)]
            finally:
                timings["ranking"] = (perf_counter() - started) * 1000.0
                record_ranking_duration(timings["ranking"] / 1000.0)

            result_status = "exact_match" if results else "not_found"
            _finalize(status=result_status, requested=requested, results_count=len(results))
            timings["total"] = sum(v for key, v in timings.items() if key != "total")

            return SearchV2Response(
                request_id=get_request_id() or str(uuid4()),
                query=query,
                status=result_status,
                requested=requested,
                results=results,
                timing_ms=timings,
            )
        except Exception:
            if result_status == "technical_failure":
                record_search_request(result_status)
                log_search_completed(
                    request_id=get_request_id(),
                    result_status=result_status,
                    results_count=0,
                    total_ms=(perf_counter() - total_started) * 1000.0,
                    deepseek_ms=timings.get("deepseek"),
                    embedding_ms=timings.get("embedding"),
                    qdrant_ms=timings.get("qdrant"),
                    ranking_ms=timings.get("ranking"),
                    query_length=len(query),
                )
            raise


__all__ = [
    "CompetitorMatch",
    "CompetitorProduct",
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
    "SearchV2Response",
]
