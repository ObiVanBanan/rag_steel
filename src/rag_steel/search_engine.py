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
    SOURCE_SCORE_FIELD_WEIGHT,
    SOURCE_SCORE_HYBRID_WEIGHT,
    SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT,
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
            "name": payload.get("name") or payload.get("steel_name"),
            "url": payload.get("url") or payload.get("steel_url"),
            "price": payload.get("price") or payload.get("price_ld"),
            "dn": payload.get("dn") or payload.get("steel_dn"),
            "pn_bar": payload.get("pn_bar") or payload.get("steel_pn_bar"),
            "connection": payload.get("connection") or payload.get("steel_connection"),
            "medium": payload.get("medium") or payload.get("steel_medium"),
            "control": payload.get("control") or payload.get("steel_control"),
        }

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

        points = self._extract_points(response)
        results = self._rank_source_points(processed, points)[: max(0, limit)]

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
