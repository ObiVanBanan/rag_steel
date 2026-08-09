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
    SOURCE_SCORE_FIELD_WEIGHT,
    SOURCE_SCORE_HYBRID_WEIGHT,
    SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT,
    get_embedding_model_spec,
    get_settings,
)
from rag_steel.normalization import (
    normalize_article,
    normalize_brand,
    normalize_connection,
    normalize_control,
    normalize_dn,
    normalize_medium,
    normalize_pn_bar,
    normalize_text,
)
from rag_steel.query_processor import EmbeddingTextAdapter, ProcessedQuery, QueryProcessor


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    relevance_rating: float | None = None
    id: str | int | None = None
    score: float | None = None
    hybrid_score: float | None = None
    product: dict[str, Any] = Field(default_factory=dict)
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
    _resolved_collection_name: str | None = field(init=False, default=None, repr=False)
    embedding_adapter: EmbeddingTextAdapter = field(init=False, repr=False)
    settings: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = None
        self._client = self.client
        self._resolved_collection_name = None
        self.settings = get_settings().for_model(self.model_name)
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

    def _query_points(
        self,
        client: QdrantClient,
        processed: ProcessedQuery,
        dense_vector: list[float],
    ) -> Any:
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
                            query=self._sparse_query(processed),
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
            text = normalize_text(value)
            if text:
                return text
        return None

    @staticmethod
    def _payload_number(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            if key.endswith("dn") or key == "dn":
                number = normalize_dn(value)
            elif key.endswith("pn") or key.endswith("pn_bar") or key == "pn_bar":
                number = normalize_pn_bar(value)
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = None
            if number is not None:
                return number
        return None

    @staticmethod
    def _normalize_hybrid_scores(points: list[dict[str, Any]]) -> list[float | None]:
        scores = [score for score in (point["raw_score"] for point in points) if score is not None]
        if not scores:
            return [None for _ in points]

        top_score = max(scores)
        if top_score <= 0:
            return [0.0 if point["raw_score"] is not None else None for point in points]

        normalized_scores: list[float | None] = []
        for point in points:
            raw_score = point["raw_score"]
            if raw_score is None:
                normalized_scores.append(None)
                continue
            normalized_scores.append(max(0.0, min(1.0, raw_score / top_score)))
        return normalized_scores

    @staticmethod
    def _text_similarity_score(query_text: str | None, source_text: str | None) -> float | None:
        query_norm = normalize_text(query_text)
        source_norm = normalize_text(source_text)
        if not query_norm or not source_norm:
            return None

        query_article = normalize_article(query_norm)
        source_article = normalize_article(source_norm)
        query_candidates = [
            query_article.article_compact,
            query_article.article_norm,
            query_norm,
        ]
        source_candidates = [
            source_article.article_compact,
            source_article.article_norm,
            source_norm,
        ]

        for query_value in query_candidates:
            if not query_value:
                continue
            query_compact = (
                normalize_article(query_value).article_compact
                or query_value.replace(" ", "")
            )
            query_normalized = normalize_text(query_value) or ""
            for source_value in source_candidates:
                if not source_value:
                    continue
                source_compact = (
                    normalize_article(source_value).article_compact
                    or source_value.replace(" ", "")
                )
                source_normalized = normalize_text(source_value) or ""

                if not query_compact or not source_compact:
                    continue
                if query_compact == source_compact or query_normalized == source_normalized:
                    return 1.0
                if query_compact.startswith(source_compact) or source_compact.startswith(
                    query_compact
                ):
                    return 0.85
                if query_compact in source_compact or source_compact in query_compact:
                    return 0.75
                if query_normalized.startswith(source_normalized) or source_normalized.startswith(
                    query_normalized
                ):
                    return 0.85
                if query_normalized in source_normalized or source_normalized in query_normalized:
                    return 0.75
        return 0.0

    def _text_exactness(self, processed: ProcessedQuery, payload: dict[str, Any]) -> float | None:
        query_candidates = [
            processed.normalized,
            processed.compact,
            processed.brand,
            *processed.possible_article_tokens,
        ]
        source_candidates = [
            self._payload_text(payload, "article", "steel_article"),
            self._payload_text(payload, "article_norm", "steel_article_norm"),
            self._payload_text(payload, "article_compact", "steel_article_compact"),
            self._payload_text(payload, "name", "steel_name"),
            self._payload_text(payload, "brand"),
        ]

        scores: list[float] = []
        for query_text in query_candidates:
            for source_text in source_candidates:
                score = self._text_similarity_score(query_text, source_text)
                if score is not None:
                    scores.append(score)
        if not scores:
            return None
        return max(scores)

    def _source_field_score(
        self,
        processed: ProcessedQuery,
        payload: dict[str, Any],
    ) -> float | None:
        comparisons: list[float] = []

        source_brand = normalize_brand(
            self._payload_text(payload, "brand", "steel_brand")
            or payload.get("brand")
            or payload.get("steel_name")
        )
        if processed.brand is not None:
            comparisons.append(1.0 if source_brand == processed.brand else 0.0)

        source_dn = self._payload_number(payload, "dn", "steel_dn")
        if processed.dn is not None:
            comparisons.append(1.0 if source_dn == processed.dn else 0.0)

        source_pn = self._payload_number(payload, "pn_bar", "steel_pn_bar")
        if processed.pn_bar is not None:
            comparisons.append(1.0 if source_pn == processed.pn_bar else 0.0)

        source_connection = normalize_connection(
            self._payload_text(payload, "connection", "steel_connection")
        )
        if processed.connection is not None:
            comparisons.append(1.0 if source_connection == processed.connection else 0.0)

        source_medium = normalize_medium(self._payload_text(payload, "medium", "steel_medium"))
        if processed.medium is not None:
            comparisons.append(1.0 if source_medium == processed.medium else 0.0)

        source_control = normalize_control(self._payload_text(payload, "control", "steel_control"))
        if processed.control is not None:
            comparisons.append(1.0 if source_control == processed.control else 0.0)

        if not comparisons:
            return None
        return sum(comparisons) / len(comparisons)

    @staticmethod
    def _weighted_average(values: list[tuple[float | None, float]]) -> float:
        weighted_total = 0.0
        total_weight = 0.0
        for value, weight in values:
            if value is None:
                continue
            weighted_total += value * weight
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return weighted_total / total_weight

    def _build_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "article": payload.get("article") or payload.get("steel_article"),
            "article_norm": payload.get("article_norm") or payload.get("steel_article_norm"),
            "name": payload.get("name") or payload.get("steel_name"),
            "url": payload.get("url") or payload.get("steel_url"),
            "price": payload.get("price") or payload.get("price_ld"),
            "dn": payload.get("dn") or payload.get("steel_dn"),
            "pn_bar": payload.get("pn_bar") or payload.get("steel_pn_bar"),
            "connection": payload.get("connection") or payload.get("steel_connection"),
            "medium": payload.get("medium") or payload.get("steel_medium"),
            "control": payload.get("control") or payload.get("steel_control"),
        }

    @staticmethod
    def _ld_product_key(product: dict[str, Any]) -> str | None:
        article_norm = product.get("article_norm")
        if article_norm:
            return str(article_norm)
        article = product.get("article")
        return str(article) if article else None

    @staticmethod
    def _ld_product_text(product: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = product.get(key)
            if value is None:
                continue
            text = normalize_text(value)
            if text:
                return text
        return None

    @staticmethod
    def _ld_product_number(product: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = product.get(key)
            if value is None:
                continue
            if key.endswith("dn") or key == "dn":
                number = normalize_dn(value)
            elif key.endswith("pn") or key.endswith("pn_bar") or key == "pn_bar":
                number = normalize_pn_bar(value)
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = None
            if number is not None:
                return number
        return None

    def _build_match_reasons(
        self,
        processed: ProcessedQuery,
        payload: dict[str, Any],
        *,
        text_exactness: float | None,
        source_field_score: float | None,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        mismatches: list[str] = []

        article = self._payload_text(payload, "article", "steel_article")
        brand = normalize_brand(self._payload_text(payload, "brand"))
        dn = self._payload_number(payload, "dn", "steel_dn")
        pn = self._payload_number(payload, "pn_bar", "steel_pn_bar")
        connection = normalize_connection(self._payload_text(payload, "connection"))
        medium = normalize_medium(self._payload_text(payload, "medium"))
        control = normalize_control(self._payload_text(payload, "control"))
        name = self._payload_text(payload, "name", "steel_name")

        if text_exactness and text_exactness >= 1.0 and article:
            reasons.append(f"Найден товар с артикулом {article}")
        elif text_exactness and text_exactness >= 0.85 and article:
            reasons.append(f"Есть сильное текстовое совпадение с артикулом {article}")
        elif text_exactness and text_exactness >= 0.75 and name:
            reasons.append(f"Есть текстовое совпадение с наименованием {name}")

        if processed.brand is not None:
            if brand == processed.brand:
                reasons.append(f"Совпадает бренд {processed.brand}")
            else:
                mismatches.append(
                    f"Бренд: ожидался {processed.brand}, найден {brand or 'не указан'}"
                )

        if processed.dn is not None:
            if dn == processed.dn:
                reasons.append(f"Совпадает DN {processed.dn:g}")
            else:
                if dn is not None:
                    mismatches.append(f"DN: ожидался {processed.dn:g}, найден {dn:g}")
                else:
                    mismatches.append(
                        f"DN: ожидался {processed.dn:g}, но поле не указано"
                    )

        if processed.pn_bar is not None:
            if pn == processed.pn_bar:
                reasons.append(f"Совпадает PN {processed.pn_bar:g}")
            else:
                if pn is not None:
                    mismatches.append(f"PN: ожидался {processed.pn_bar:g}, найден {pn:g}")
                else:
                    mismatches.append(
                        f"PN: ожидался {processed.pn_bar:g}, но поле не указано"
                    )

        if processed.connection is not None:
            if connection == processed.connection:
                reasons.append(f"Совпадает присоединение {processed.connection}")
            else:
                mismatches.append(
                    "Присоединение: ожидалось "
                    f"{processed.connection}, найден {connection or 'не указано'}"
                )

        if processed.medium is not None:
            if medium == processed.medium:
                reasons.append(f"Совпадает среда {processed.medium}")
            else:
                mismatches.append(
                    f"Среда: ожидалась {processed.medium}, найден {medium or 'не указана'}"
                )

        if processed.control is not None:
            if control == processed.control:
                reasons.append(f"Совпадает управление {processed.control}")
            else:
                mismatches.append(
                    f"Управление: ожидалось {processed.control}, найдено {control or 'не указано'}"
                )

        if source_field_score is not None and source_field_score < 1.0 and not mismatches:
            mismatches.append("Часть структурных полей не совпадает")

        return reasons, mismatches

    def _ld_field_score(self, processed: ProcessedQuery, product: dict[str, Any]) -> float | None:
        comparisons: list[float] = []

        if processed.dn is not None:
            ld_dn = self._ld_product_number(product, "dn")
            comparisons.append(1.0 if ld_dn == processed.dn else 0.0)

        if processed.pn_bar is not None:
            ld_pn = self._ld_product_number(product, "pn_bar")
            if ld_pn is None:
                comparisons.append(0.0)
            elif ld_pn == processed.pn_bar:
                comparisons.append(1.0)
            elif ld_pn > processed.pn_bar:
                comparisons.append(0.85)
            else:
                comparisons.append(0.0)

        if processed.connection is not None:
            ld_connection = normalize_connection(
                self._ld_product_text(product, "connection")
            )
            comparisons.append(1.0 if ld_connection == processed.connection else 0.0)

        if processed.medium is not None:
            ld_medium = normalize_medium(self._ld_product_text(product, "medium"))
            comparisons.append(1.0 if ld_medium == processed.medium else 0.0)

        if processed.control is not None:
            ld_control = normalize_control(self._ld_product_text(product, "control"))
            comparisons.append(1.0 if ld_control == processed.control else 0.0)

        if not comparisons:
            return None
        return sum(comparisons) / len(comparisons)

    def _build_ld_match_reasons(
        self,
        processed: ProcessedQuery,
        product: dict[str, Any],
        *,
        ld_field_score: float | None,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        mismatches: list[str] = []

        article = self._ld_product_text(product, "article", "article_norm")
        dn = self._ld_product_number(product, "dn")
        pn = self._ld_product_number(product, "pn_bar")
        connection = normalize_connection(self._ld_product_text(product, "connection"))
        medium = normalize_medium(self._ld_product_text(product, "medium"))
        control = normalize_control(self._ld_product_text(product, "control"))

        if processed.dn is not None:
            if dn == processed.dn:
                reasons.append(f"Совпадает DN {processed.dn:g}")
            else:
                if dn is not None:
                    mismatches.append(f"DN: ожидался {processed.dn:g}, найден {dn:g}")
                else:
                    mismatches.append(
                        f"DN: ожидался {processed.dn:g}, но поле не указано"
                    )

        if processed.pn_bar is not None:
            if pn == processed.pn_bar:
                reasons.append(f"Совпадает PN {processed.pn_bar:g}")
            elif pn is not None and pn > processed.pn_bar:
                reasons.append(f"PN {pn:g} выше запрошенного {processed.pn_bar:g}")
            else:
                if pn is not None:
                    mismatches.append(f"PN: ожидался {processed.pn_bar:g}, найден {pn:g}")
                else:
                    mismatches.append(
                        f"PN: ожидался {processed.pn_bar:g}, но поле не указано"
                    )

        if processed.connection is not None:
            if connection == processed.connection:
                reasons.append(f"Совпадает присоединение {processed.connection}")
            else:
                mismatches.append(
                    "Присоединение: ожидалось "
                    f"{processed.connection}, найден {connection or 'не указано'}"
                )

        if processed.medium is not None:
            if medium == processed.medium:
                reasons.append(f"Совпадает среда {processed.medium}")
            else:
                mismatches.append(
                    f"Среда: ожидалась {processed.medium}, найден {medium or 'не указана'}"
                )

        if processed.control is not None:
            if control == processed.control:
                reasons.append(f"Совпадает управление {processed.control}")
            else:
                mismatches.append(
                    f"Управление: ожидалось {processed.control}, найдено {control or 'не указано'}"
                )

        if ld_field_score is None and not reasons and article:
            reasons.append(f"LD-кандидат {article}")

        return reasons, mismatches

    @staticmethod
    def _append_evidence(
        evidence_by_key: dict[str, dict[str, Any]],
        *,
        source_article: str | None,
        source_name: str | None,
        source_score: float | None,
        source_rank: int,
    ) -> None:
        key = source_article or source_name or str(source_rank)
        current = evidence_by_key.get(key)
        evidence = {
            "source_article": source_article,
            "source_name": source_name,
            "source_score": source_score,
        }
        if current is None or (source_score is not None and source_score > current["source_score"]):
            evidence["source_rank"] = source_rank
            evidence_by_key[key] = evidence

    def _rank_ld_candidates(
        self,
        processed: ProcessedQuery,
        source_results: list[SearchResult],
    ) -> list[SearchResult]:
        ranked_candidates: list[dict[str, Any]] = []

        for source_result in source_results:
            payload = source_result.payload
            source_score = source_result.score_breakdown.get("source_score")
            source_article = source_result.product.get("article")
            source_name = source_result.product.get("name")
            ld_candidates = payload.get("ld_candidates") or []
            for candidate in ld_candidates:
                if hasattr(candidate, "model_dump"):
                    candidate = candidate.model_dump(mode="json")
                product = {
                    "article": candidate.get("article") or candidate.get("ld_article"),
                    "article_norm": candidate.get("article_norm")
                    or candidate.get("ld_article_norm"),
                    "name": candidate.get("name") or candidate.get("ld_name"),
                    "url": candidate.get("url") or candidate.get("ld_url"),
                    "price": candidate.get("price") or candidate.get("price_ld"),
                    "dn": candidate.get("dn") or candidate.get("ld_dn"),
                    "pn_bar": candidate.get("pn_bar") or candidate.get("ld_pn_mpa"),
                    "connection": candidate.get("connection")
                    or candidate.get("ld_connection"),
                    "medium": candidate.get("medium") or candidate.get("ld_medium"),
                    "control": candidate.get("control") or candidate.get("ld_control"),
                }
                ld_field_score = self._ld_field_score(processed, product)
                if source_score is None:
                    continue
                final_score = source_score
                if ld_field_score is not None:
                    final_score = (source_score * 0.70) + (ld_field_score * 0.30)

                ld_match_reasons, ld_mismatches = self._build_ld_match_reasons(
                    processed,
                    product,
                    ld_field_score=ld_field_score,
                )
                score_breakdown = dict(source_result.score_breakdown)
                if ld_field_score is not None:
                    score_breakdown["ld_field_score"] = ld_field_score
                score_breakdown["final_score"] = final_score

                ranked_candidates.append(
                    {
                        "dedupe_key": self._ld_product_key(product),
                        "product": product,
                        "source_evidence": {
                            "source_article": source_article,
                            "source_name": source_name,
                            "source_score": source_score,
                            "source_rank": source_result.rank,
                        },
                        "match_reasons": ld_match_reasons,
                        "mismatches": ld_mismatches,
                        "score_breakdown": score_breakdown,
                        "source_score": source_score,
                        "final_score": final_score,
                        "hybrid_score": source_result.hybrid_score or 0.0,
                    }
                )

        deduplicated: dict[str, dict[str, Any]] = {}
        for candidate in ranked_candidates:
            dedupe_key = candidate["dedupe_key"]
            if dedupe_key is None:
                continue
            current = deduplicated.get(dedupe_key)
            if current is None:
                candidate["evidence_by_key"] = {}
                self._append_evidence(
                    candidate["evidence_by_key"],
                    **candidate["source_evidence"],
                )
                deduplicated[dedupe_key] = candidate
                continue

            current_score = current["final_score"]
            if candidate["final_score"] > current_score:
                candidate["evidence_by_key"] = current.get("evidence_by_key", {}).copy()
                self._append_evidence(
                    candidate["evidence_by_key"],
                    **candidate["source_evidence"],
                )
                deduplicated[dedupe_key] = candidate
            else:
                evidence_by_key = current.setdefault("evidence_by_key", {})
                self._append_evidence(
                    evidence_by_key,
                    **candidate["source_evidence"],
                )

        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (-item["final_score"], -item["source_score"], str(item["dedupe_key"])),
        )

        results: list[SearchResult] = []
        for rank, candidate in enumerate(ordered, start=1):
            final_score = candidate["final_score"]
            results.append(
                SearchResult(
                    rank=rank,
                    relevance_rating=round(final_score * 100, 1),
                    id=candidate["dedupe_key"],
                    score=final_score,
                    hybrid_score=candidate["hybrid_score"],
                    product=candidate["product"],
                    payload={},
                    source_evidence=list(candidate.get("evidence_by_key", {}).values())[:3],
                    match_reasons=candidate["match_reasons"],
                    mismatches=candidate["mismatches"],
                    score_breakdown=candidate["score_breakdown"],
                )
            )
        return results

    def _rank_source_points(
        self,
        processed: ProcessedQuery,
        points: list[Any],
    ) -> list[SearchResult]:
        raw_points: list[dict[str, Any]] = []
        for point in points:
            payload = self._extract_payload(point)
            raw_points.append(
                {
                    "point": point,
                    "id": self._extract_id(point),
                    "raw_score": self._extract_score(point),
                    "payload": payload,
                }
            )

        normalized_hybrid_scores = self._normalize_hybrid_scores(raw_points)
        ranked_candidates: list[dict[str, Any]] = []
        for index, item in enumerate(raw_points):
            payload = item["payload"]
            hybrid_score = normalized_hybrid_scores[index]
            text_exactness = self._text_exactness(processed, payload)
            source_field_score = self._source_field_score(processed, payload)
            source_score = self._weighted_average(
                [
                    (hybrid_score, SOURCE_SCORE_HYBRID_WEIGHT),
                    (text_exactness, SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT),
                    (source_field_score, SOURCE_SCORE_FIELD_WEIGHT),
                ]
            )
            match_reasons, mismatches = self._build_match_reasons(
                processed,
                payload,
                text_exactness=text_exactness,
                source_field_score=source_field_score,
            )
            product = self._build_product(payload)
            score_breakdown: dict[str, float] = {"hybrid_score": hybrid_score or 0.0}
            if text_exactness is not None:
                score_breakdown["text_exactness"] = text_exactness
            if source_field_score is not None:
                score_breakdown["source_field_score"] = source_field_score
            score_breakdown["source_score"] = source_score

            ranked_candidates.append(
                {
                    "dedupe_key": str(
                        payload.get("steel_id") or item["id"] or product.get("article")
                    ),
                    "result": SearchResult(
                        rank=0,
                        id=item["id"],
                        score=item["raw_score"],
                        hybrid_score=hybrid_score,
                        product=product,
                        payload=payload,
                        source_evidence=[
                            {
                                "article": product.get("article"),
                                "name": product.get("name"),
                            }
                        ],
                        match_reasons=match_reasons,
                        mismatches=mismatches,
                        score_breakdown=score_breakdown,
                    ),
                    "source_score": source_score,
                    "hybrid_score": hybrid_score or 0.0,
                    "text_exactness": text_exactness or 0.0,
                    "id_key": str(item["id"] or ""),
                }
            )

        deduplicated: dict[str, dict[str, Any]] = {}
        for candidate in ranked_candidates:
            key = candidate["dedupe_key"]
            current = deduplicated.get(key)
            if current is None or candidate["source_score"] > current["source_score"]:
                deduplicated[key] = candidate

        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                -item["source_score"],
                -item["hybrid_score"],
                -item["text_exactness"],
                item["id_key"],
            ),
        )

        results: list[SearchResult] = []
        for rank, candidate in enumerate(ordered, start=1):
            result = candidate["result"]
            results.append(result.model_copy(update={"rank": rank}))
        return results

    def _dense_query_text(self, processed: ProcessedQuery) -> str:
        return self.embedding_adapter.prepare_query(processed.semantic_text)

    def _apply_result_threshold(self, results: list[SearchResult]) -> list[SearchResult]:
        threshold = self.settings.result_score_threshold
        filtered = [
            result for result in results if result.score is not None and result.score >= threshold
        ]
        for rank, result in enumerate(filtered, start=1):
            result.rank = rank
        return filtered

    def _sparse_query(self, processed: ProcessedQuery) -> models.Document:
        return models.Document(
            text=processed.lexical_text,
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
        return self._coerce_vector(
            vectors
        )

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
        processed = self.query_processor.process(query)
        timings["normalize"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        dense_query_text = self._dense_query_text(processed)
        dense_vector = self._encode_query(dense_query_text)
        timings["embedding"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        response = self._query_points(self._get_client(), processed, dense_vector)
        timings["qdrant"] = (perf_counter() - started) * 1000.0

        points = self._extract_points(response)
        source_results = self._rank_source_points(processed, points)

        started = perf_counter()
        results = self._rank_ld_candidates(processed, source_results)[: max(0, limit)]
        results = self._apply_result_threshold(results)
        timings["ranking"] = (perf_counter() - started) * 1000.0

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
