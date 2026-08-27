from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

import rag_steel.search_engine as search_engine_mod
from rag_steel.index_metadata import SUPPORTED_INDEX_FORMAT_VERSION, check_index_compatibility
from rag_steel.normalization import normalize_connection, normalize_text
from rag_steel.observability import reset_request_id, set_request_id
from rag_steel.query_constraints import QueryConstraints, extract_query_constraints
from rag_steel.runtime import EmbeddingTimeoutError, SearchBackendTimeoutError
from rag_steel.search_engine import SearchEngine, SearchResponse
from rag_steel.search_messages import SEARCH_FAILURE_MESSAGE
from rag_steel.settings import Settings


@dataclass(slots=True)
class FakeEmbedder:
    calls: list[dict[str, object]]
    model_name: str = "fake"
    dimension: int = 3
    embedding_revision: str = ""

    def embed_query(self, text: str) -> list[float]:
        self.calls.append({"kind": "query", "texts": [text]})
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append({"kind": "documents", "texts": list(texts)})
        return [[0.1, 0.2, 0.3] for _ in texts]


class FailingEmbedder(FakeEmbedder):
    def embed_query(self, text: str) -> list[float]:
        self.calls.append({"kind": "query", "texts": [text]})
        raise EmbeddingTimeoutError("timed out")


class FakeAttributeExtractor:
    def extract(self, query: str) -> search_engine_mod.ExtractedAttributes:
        constraints = extract_query_constraints(query)
        payload = constraints.model_dump()
        payload["raw_brand"] = payload.pop("brand", None)
        payload["article"] = None
        return search_engine_mod.ExtractedAttributes.model_validate(payload)


class StaticAttributeExtractor:
    def __init__(self, **attributes: object) -> None:
        self.attributes = attributes

    def extract(self, query: str) -> search_engine_mod.ExtractedAttributes:
        del query
        payload = {
            "raw_brand": None,
            "article": None,
            "dn": None,
            "pn_bar": None,
            "connection": None,
            "body_material": None,
            "medium": None,
            "control": None,
            "temperature": None,
            "length_mm": None,
            "series": None,
            **self.attributes,
        }
        return search_engine_mod.ExtractedAttributes.model_validate(payload)


class NoBrandFallbackExtractor:
    def extract(self, query: str) -> search_engine_mod.ExtractedAttributes:
        del query
        return search_engine_mod.ExtractedAttributes.model_validate(
            {
                "raw_brand": None,
                "article": None,
                "dn": 20,
                "pn_bar": 40,
                "connection": "фланцевое",
                "body_material": None,
                "medium": None,
                "control": None,
                "temperature": None,
                "length_mm": None,
                "series": None,
            }
        )


class RawArticleNotFoundExtractor:
    def extract(self, query: str) -> search_engine_mod.ExtractedAttributes:
        return search_engine_mod.ExtractedAttributes.model_validate(
            {
                "raw_brand": None,
                "article": "107-5529ШШ",
                "dn": None,
                "pn_bar": None,
                "connection": None,
                "body_material": None,
                "medium": None,
                "control": None,
                "temperature": None,
                "length_mm": None,
                "series": None,
            }
        )


class DroppedHardConstraintExtractor:
    def __init__(self, *, brand: str = "Temper") -> None:
        self.brand = brand

    def extract(self, query: str) -> search_engine_mod.ExtractedAttributes:
        del query
        return search_engine_mod.ExtractedAttributes.model_validate(
            {
                "raw_brand": self.brand,
                "article": None,
                "dn": None,
                "pn_bar": None,
                "connection": None,
                "body_material": None,
                "medium": None,
                "control": None,
                "temperature": None,
                "length_mm": None,
                "series": None,
            }
        )


def _ld_candidate(
    article: str,
    article_norm: str,
    *,
    name: str,
    dn: float,
    pn_bar: float,
    connection: str,
    medium: str,
    control: str,
    url: str,
    price: float,
) -> dict[str, object]:
    return {
        "article": article,
        "article_norm": article_norm,
        "name": name,
        "url": url,
        "dn": dn,
        "pn_bar": pn_bar,
        "connection": connection,
        "medium": medium,
        "control": control,
        "price": price,
    }


class FakeQdrantClient:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="doc-1",
                    score=0.91,
                    payload={
                        "article": "1184399",
                        "article_norm": "1184399",
                        "name": "Temper DN80 PN16",
                        "ld_candidates": [
                            _ld_candidate(
                                "11100800162MULD000003000",
                                "11100800162muld000003000",
                                name="LD Temper DN80 PN16",
                                dn=80,
                                pn_bar=16,
                                connection="flanged",
                                medium="liquid",
                                control="manual",
                                url="https://example.invalid/ld-a",
                                price=12130,
                            ),
                            _ld_candidate(
                                "11100800162MULD000004000",
                                "11100800162muld000004000",
                                name="LD Temper DN50 PN16",
                                dn=50,
                                pn_bar=16,
                                connection="flanged",
                                medium="liquid",
                                control="manual",
                                url="https://example.invalid/ld-b",
                                price=9800,
                            ),
                        ],
                    },
                ),
                SimpleNamespace(
                    id="doc-2",
                    score=0.97,
                    payload={
                        "article": "a0486",
                        "article_norm": "a0486",
                        "name": "Broen DN50 PN10",
                        "brand": "Broen",
                        "ld_candidates": [
                            _ld_candidate(
                                "11100800162MULD000003000",
                                "11100800162muld000003000",
                                name="LD Temper DN80 PN16",
                                dn=80,
                                pn_bar=16,
                                connection="flanged",
                                medium="liquid",
                                control="manual",
                                url="https://example.invalid/ld-a",
                                price=12130,
                            ),
                            _ld_candidate(
                                "11100800162MULD000005000",
                                "11100800162muld000005000",
                                name="LD Broen DN50 PN10",
                                dn=50,
                                pn_bar=10,
                                connection="threaded",
                                medium="gas",
                                control="electric",
                                url="https://example.invalid/ld-c",
                                price=15400,
                            ),
                        ],
                    },
                ),
            ]
        )


class RegressionQdrantClient(FakeQdrantClient):
    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        sparse_query = kwargs["prefetch"][1].query.text  # type: ignore[index]
        normalized = normalize_text(sparse_query) or ""

        if not normalized or "nonexistent" in normalized or "empty query" in normalized:
            return SimpleNamespace(points=[])

        result_count = (
            20
            if any(token in normalized for token in ["temper", "broen", "dn80 pn16", "flanged"])
            else 7
        )

        points = []
        for index in range(result_count):
            candidate_article = f"ld-{index:02d}"
            points.append(
                SimpleNamespace(
                    id=f"source-{index}",
                    score=1.0 - (index * 0.01),
                    payload={
                        "article": f"SOURCE-{index}",
                        "article_norm": f"source-{index}",
                        "name": f"Source {index}",
                        "ld_candidates": [
                            _ld_candidate(
                                candidate_article,
                                candidate_article,
                                name=f"LD {index}",
                                dn=80 if result_count == 20 else 50,
                                pn_bar=16,
                                connection="flanged",
                                medium="liquid",
                                control="manual",
                                url=f"https://example.invalid/ld-{index}",
                                price=1000 + index,
                            )
                        ],
                    },
                )
            )
        return SimpleNamespace(points=points)


class MissingAliasQdrantClient(FakeQdrantClient):
    def __init__(self) -> None:
        super().__init__()
        self.get_collection_calls: list[str] = []

    @staticmethod
    def _missing_collection(name: str) -> UnexpectedResponse:
        return UnexpectedResponse(
            404,
            "Not Found",
            (
                '{"status":{"error":"Not found: Collection '
                f"`{name}` doesn't exist!"
                '"},"time":0.0001}'
            ).encode("utf-8"),
            Headers(),
        )

    def get_collection(self, *, collection_name: str, **_: object) -> object:
        self.get_collection_calls.append(collection_name)
        raise self._missing_collection(collection_name)

    def count(self, *, collection_name: str, **_: object) -> object:
        del collection_name
        return SimpleNamespace(count=7)

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        if kwargs["collection_name"] == "steel_products_active":
            raise self._missing_collection("steel_products_active")
        return super().query_points(**kwargs)


class EmptyQdrantClient:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=[])


class BrandFallbackResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def resolve(
        self,
        *,
        raw_brand: object,
        raw_article: object,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> SimpleNamespace:
        call = {
            "raw_brand": raw_brand,
            "raw_article": raw_article,
            "dn": dn,
            "pn_bar": pn_bar,
            "connection": connection,
        }
        self.calls.append(call)
        if raw_brand == "Temper":
            return SimpleNamespace(
                raw_brand="Temper",
                raw_article=raw_article,
                brand=SimpleNamespace(raw="Temper", canonical="Temper", match_type="exact"),
                article=None,
                resolution_mode="brand_exact",
                reason_code=None,
            )
        return SimpleNamespace(
            raw_brand=None,
            raw_article=raw_article,
            brand=SimpleNamespace(raw=None, canonical=None, match_type=None),
            article=None,
            resolution_mode="no_identity",
            reason_code="COMPETITOR_BRAND_REQUIRED",
        )


class ArticleNotFoundResolver:
    def resolve(
        self,
        *,
        raw_brand: object,
        raw_article: object,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> SimpleNamespace:
        del raw_brand, dn, pn_bar, connection
        return SimpleNamespace(
            raw_brand=None,
            raw_article=raw_article,
            brand=SimpleNamespace(raw=None, canonical=None, match_type=None),
            article=SimpleNamespace(
                raw=raw_article,
                normalized=str(raw_article).casefold() if raw_article is not None else None,
                compact=str(raw_article).casefold() if raw_article is not None else None,
                article=None,
                brand=None,
                match_type=None,
                reason_code="ARTICLE_NOT_FOUND",
            ),
            resolution_mode="article_not_found",
            reason_code="ARTICLE_NOT_FOUND",
        )


class ExactArticleResolver:
    def __init__(self, source_product: dict[str, object], *, brand: str = "Stout") -> None:
        self.source_product = source_product
        self.brand = brand
        self.calls: list[dict[str, object]] = []

    def resolve(
        self,
        *,
        raw_brand: object,
        raw_article: object,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "raw_brand": raw_brand,
                "raw_article": raw_article,
                "dn": dn,
                "pn_bar": pn_bar,
                "connection": connection,
            }
        )
        article = str(raw_article or self.source_product.get("article") or "ART-1")
        return SimpleNamespace(
            raw_brand=raw_brand,
            raw_article=raw_article,
            brand=SimpleNamespace(raw=raw_brand, canonical=self.brand, match_type="exact"),
            article=SimpleNamespace(
                raw=article,
                normalized=article.casefold(),
                compact=article.casefold(),
                article=article,
                brand=self.brand,
                match_type="exact",
                reason_code=None,
                source_product=self.source_product,
            ),
            resolution_mode="article_exact",
            reason_code=None,
        )


class AliasedQdrantClient(FakeQdrantClient):
    def __init__(self) -> None:
        super().__init__()
        self.get_collection_calls: list[str] = []

    def get_collection(self, *, collection_name: str, **_: object) -> object:
        self.get_collection_calls.append(collection_name)
        return SimpleNamespace(
            metadata={
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_revision": "",
                "embedding_dimension": 3,
            },
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=3)})
            ),
        )

    def count(self, *, collection_name: str, **_: object) -> object:
        return SimpleNamespace(count=7)

    def get_aliases(self, **_: object) -> object:
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(
                    alias_name="steel_products_active",
                    collection_name="steel_products_20260817T010203Z",
                )
            ]
        )


class HotAliasSwitchQdrantClient(FakeQdrantClient):
    def __init__(self) -> None:
        super().__init__()
        self.get_collection_calls: list[str] = []
        self.query_calls: list[str] = []
        self._alias_target = "collection_A"

    def get_collection(self, *, collection_name: str, **_: object) -> object:
        self.get_collection_calls.append(collection_name)
        if collection_name != "steel_products_active":
            raise MissingAliasQdrantClient._missing_collection(collection_name)
        return SimpleNamespace(
            metadata={
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=3)})
            ),
        )

    def count(self, *, collection_name: str, **_: object) -> object:
        return SimpleNamespace(count=7)

    def get_aliases(self, **_: object) -> object:
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(
                    alias_name="steel_products_active",
                    collection_name=self._alias_target,
                )
            ]
        )

    def query_points(self, **kwargs: object) -> object:
        collection_name = str(kwargs["collection_name"])
        self.query_calls.append(collection_name)
        if collection_name == "collection_A":
            raise MissingAliasQdrantClient._missing_collection(collection_name)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="hit-1",
                    score=0.91,
                    payload={
                        "article": "1184399",
                        "article_norm": "1184399",
                        "name": "Temper DN80 PN16",
                        "ld_candidates": [
                            _ld_candidate(
                                "11100800162MULD000003000",
                                "11100800162muld000003000",
                                name="LD Temper DN80 PN16",
                                dn=80,
                                pn_bar=16,
                                connection="flanged",
                                medium="liquid",
                                control="manual",
                                url="https://example.invalid/ld-a",
                                price=12130,
                            )
                        ],
                    },
                )
            ]
        )


class V2QdrantClient:
    def __init__(self, points: list[object]) -> None:
        self.points = points
        self.query_calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def _enable_search_trace(engine: SearchEngine) -> None:
    engine.settings = replace(engine.settings, search_trace_enabled=True)


def _search_trace_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for record in caplog.records:
        payload = json.loads(record.message)
        if payload.get("event") == "search_trace":
            events.append(payload)
    return events


def _logged_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(record.message) for record in caplog.records]


def _v2_ld_candidate() -> dict[str, object]:
    return _ld_candidate(
        "11100800162MULD000003000",
        "11100800162muld000003000",
        name="LD Temper DN80 PN16",
        dn=80,
        pn_bar=16,
        connection="flanged",
        medium="liquid",
        control="manual",
        url="https://example.invalid/ld-a",
        price=12130,
    )


def _v2_source_point(
    *,
    article: str = "1184399",
    score: float = 0.91,
    brand: str | None = "Temper",
    dn: float | None = 80,
    pn_bar: float | None = 16,
    connection: str | None = "flanged",
    body_material: str | None = "сталь 09г2с",
    medium: str | None = None,
    control: str | None = None,
    temperature: str | None = None,
    length_mm: float | None = None,
    name: str = "Temper DN80 PN16",
) -> object:
    payload: dict[str, object] = {
        "article": article,
        "article_norm": article.casefold(),
        "name": name,
        "ld_candidates": [_v2_ld_candidate()],
    }
    if brand is not None:
        payload["brand"] = brand
    if dn is not None:
        payload["dn"] = dn
    if pn_bar is not None:
        payload["pn_bar"] = pn_bar
    if connection is not None:
        payload["connection"] = connection
    if body_material is not None:
        payload["body_material"] = body_material
    if medium is not None:
        payload["medium"] = medium
    if control is not None:
        payload["control"] = control
    if temperature is not None:
        payload["temperature"] = temperature
    if length_mm is not None:
        payload["length_mm"] = length_mm
    return SimpleNamespace(id=f"doc-{article}", score=score, payload=payload)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Temper DN80 PN16", {"brand": "Temper", "dn": 80, "pn_bar": 16}),
        ("Temper DN80/PN16", {"dn": 80, "pn_bar": 16}),
        ("Temper DN80PN16", {"dn": 80, "pn_bar": 16}),
        ("Temper Ду80 Ру16", {"dn": 80, "pn_bar": 16}),
        ("Temper Ду80Ру16", {"dn": 80, "pn_bar": 16}),
        ("Темпер серия 60", {"series": "60"}),
        ("Temper series 60", {"series": "60"}),
    ],
)
def test_extract_query_constraints_parses_compact_filters(
    query: str, expected: dict[str, object]
) -> None:
    constraints = extract_query_constraints(query)
    for key, value in expected.items():
        assert getattr(constraints, key) == value


def test_extract_query_constraints_unifies_connection_synonyms() -> None:
    expected = normalize_connection("под приварку")
    assert extract_query_constraints("сварной").connection == expected
    assert extract_query_constraints("под приварку").connection == expected
    assert extract_query_constraints("приварное").connection == expected
    assert extract_query_constraints("приварной").connection == expected
    assert extract_query_constraints("welded").connection == expected


def test_extract_query_constraints_normalizes_brass_material() -> None:
    assert extract_query_constraints("Valtec кран латунный DN20 PN40").body_material == "латунь"


@pytest.mark.parametrize(
    ("query", "expected_brand"),
    [
        ("ADL DN80 PN16", "ADL"),
        ("брон DN80 PN16", "Broen"),
        ("алсо DN80 PN16", "ALSO"),
    ],
)
def test_extract_query_constraints_preserves_brand_aliases(query: str, expected_brand: str) -> None:
    assert extract_query_constraints(query).brand == expected_brand


@pytest.mark.parametrize(
    (
        "query",
        "source_kwargs",
        "expected_status",
        "expected_results",
    ),
    [
        ("Temper DN80 PN16", {}, "exact_match", 1),
        ("Temper DN80 PN16", {"pn_bar": 25}, "exact_match", 1),
        ("Temper DN80 PN25", {}, "not_found", 0),
        ("Temper DN80", {"dn": None}, "not_found", 0),
        ("Temper DN80 PN16 сталь 09Г2С", {"body_material": "09Г2С"}, "exact_match", 1),
        ("Temper DN80 PN16 сталь 09Г2С", {"body_material": None}, "not_found", 0),
        ("Broen DN80 PN16", {"brand": "Broen"}, "exact_match", 1),
        ("Temper DN80 PN16", {"brand": "Broen"}, "not_found", 0),
        ("Temper DN50 PN16", {}, "not_found", 0),
        ("Temper DN80 PN16 сталь 20", {"body_material": "сталь 09Г2С"}, "not_found", 0),
    ],
)
def test_search_v2_is_strict_and_does_not_fallback(
    query: str,
    source_kwargs: dict[str, object],
    expected_status: str,
    expected_results: int,
) -> None:
    point = _v2_source_point(**source_kwargs)
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = V2QdrantClient([point])
    engine = SearchEngine(
        embedder=fake_embedder,
        client=fake_client,
        attribute_extractor=FakeAttributeExtractor(),
    )

    response = engine.search_v2(query, limit=5)

    assert response.status == expected_status
    assert len(response.results) == expected_results
    if expected_results:
        expected_pn = source_kwargs.get("pn_bar", 16)
        assert response.results[0].match_type == "exact_match"
        assert response.results[0].differences == {}
        assert response.results[0].competitor.article == "1184399"
        assert response.results[0].competitor.dn == 80
        assert response.results[0].competitor.pn_bar == expected_pn
    assert fake_client.query_calls


def test_search_v2_builds_minimum_pressure_qdrant_filter() -> None:
    filter_ = SearchEngine._build_query_filter(
        QueryConstraints(
            brand="Temper",
            dn=80,
            pn_bar=16,
            connection="flanged",
        )
    )

    assert filter_ is not None
    pn_condition = next(condition for condition in filter_.must if condition.key == "pn_bar")
    assert pn_condition.range.gte == 16.0
    assert pn_condition.range.lte is None


def test_search_v2_adds_stable_attributes_to_qdrant_filter_without_length_or_series() -> None:
    filter_ = SearchEngine._build_query_filter(
        QueryConstraints(
            brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="фланцевое",
            body_material="сталь 20",
            medium="газ",
            control="электропривод",
            length_mm=180,
            series="3",
        )
    )

    assert filter_ is not None
    conditions = {condition.key: condition for condition in filter_.must}
    assert conditions["body_material"].match.value == "сталь 20"
    assert conditions["medium"].match.value == "газ"
    assert conditions["control"].match.value == "электропривод"
    assert "length_mm" not in conditions
    assert "series" not in conditions


def test_search_v2_matches_connection_synonyms_exactly() -> None:
    point = _v2_source_point(connection="welded")
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient([point]),
        attribute_extractor=FakeAttributeExtractor(),
    )

    response = engine.search_v2("Temper DN80 PN16 welded", limit=5)

    assert response.status == "exact_match"
    assert len(response.results) == 1
    assert response.results[0].competitor.connection == "welded"


def test_search_v2_keeps_brass_material_in_retrieval_query() -> None:
    point = _v2_source_point(
        brand="Valtec",
        dn=20,
        pn_bar=40,
        connection=None,
        body_material="латунь",
        name="VALTEC кран шаровой латунный DN20 PN40",
    )
    fake_embedder = FakeEmbedder(calls=[])
    engine = SearchEngine(
        embedder=fake_embedder,
        client=V2QdrantClient([point]),
        attribute_extractor=FakeAttributeExtractor(),
    )

    response = engine.search_v2("Valtec кран латунный DN20 PN40", limit=5)

    assert response.status == "exact_match"
    assert response.results[0].competitor.body_material == "латунь"
    assert "латунь" in str(fake_embedder.calls[0]["texts"][0])


def test_search_v2_rejects_candidate_with_different_deepseek_material() -> None:
    points = [
        _v2_source_point(
            article="brass",
            brand="Stout",
            dn=20,
            body_material="латунь",
            name="Stout DN20 латунь",
        ),
        _v2_source_point(
            article="steel",
            brand="Stout",
            dn=20,
            body_material="сталь 20",
            name="Stout DN20 сталь",
        ),
    ]
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient(points),
        attribute_extractor=StaticAttributeExtractor(
            raw_brand="Stout",
            dn=20,
            body_material="латунь",
        ),
    )

    response = engine.search_v2("Stout DN20 латунный", limit=5)

    assert response.status == "exact_match"
    assert [result.competitor.article for result in response.results] == ["brass"]


@pytest.mark.parametrize(
    ("constraints", "source_kwargs", "expected"),
    [
        (QueryConstraints(brand="PALUR"), {"brand": "MARSHAL"}, False),
        (QueryConstraints(dn=500), {"dn": 400}, False),
        (QueryConstraints(pn_bar=16), {"pn_bar": 16}, True),
        (QueryConstraints(pn_bar=16), {"pn_bar": 25}, True),
        (QueryConstraints(pn_bar=16), {"pn_bar": 10}, False),
        (QueryConstraints(connection="фланцевое"), {"connection": "сварное"}, False),
        (QueryConstraints(body_material="латунь"), {"body_material": "латунь"}, True),
        (QueryConstraints(body_material="латунь"), {"body_material": "сталь 20"}, False),
        (QueryConstraints(body_material="латунь"), {"body_material": None}, False),
        (QueryConstraints(medium="газ"), {"medium": "газ"}, True),
        (QueryConstraints(medium="газ"), {"medium": "жидкость"}, False),
        (QueryConstraints(control="электропривод"), {"control": "электропривод"}, True),
        (QueryConstraints(control="электропривод"), {"control": "ручное"}, False),
        (QueryConstraints(series="3"), {"name": "PALUR series 3 DN500"}, True),
        (QueryConstraints(series="3"), {"name": "PALUR series 30 DN500"}, False),
        (QueryConstraints(temperature="150"), {"temperature": "-40...200"}, True),
        (QueryConstraints(temperature="250"), {"temperature": "-40...200"}, False),
    ],
)
def test_source_product_matches_attribute_constraints(
    constraints: QueryConstraints,
    source_kwargs: dict[str, object],
    expected: bool,
) -> None:
    point = _v2_source_point(**source_kwargs)
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient([point]),
        attribute_extractor=FakeAttributeExtractor(),
    )

    payload = engine._extract_payload(point)
    source_product = engine._build_source_product(payload)

    assert engine._source_product_matches_constraints(source_product, constraints) is expected


def test_temperature_range_can_cover_requested_interval() -> None:
    assert SearchEngine._temperature_interval("-60...-20") == (-60.0, -20.0)
    assert SearchEngine._temperature_matches("-20...150", "-40...200")
    assert SearchEngine._temperature_matches("150", "до 200")
    assert SearchEngine._temperature_matches("-30", "-60...-20")
    assert not SearchEngine._temperature_matches("-20...250", "-40...200")
    assert not SearchEngine._temperature_matches("-10", "-60...-20")


def test_search_v2_reranks_all_candidates_by_nearest_length_before_limit() -> None:
    points = [
        _v2_source_point(article="390", brand="MARSHAL", length_mm=390, score=0.99),
        _v2_source_point(article="190", brand="MARSHAL", length_mm=190, score=0.1),
        _v2_source_point(article="180", brand="MARSHAL", length_mm=180, score=0.05),
    ]
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient(points),
        attribute_extractor=StaticAttributeExtractor(raw_brand="MARSHAL", length_mm=180),
    )

    response = engine.search_v2("Подбери кран MARSHAL в строительную длину 180", limit=2)

    assert [result.competitor.article for result in response.results] == ["180", "190"]
    assert response.results[0].differences == {}
    assert response.results[1].differences == {
        "length_mm": {
            "requested": 180.0,
            "actual": 190.0,
            "delta": 10.0,
        }
    }


def test_search_v2_trace_logs_request_path_with_normalized_attributes_and_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")
    token = set_request_id("trace-request-1")
    try:
        points = [
            _v2_source_point(article="390", brand="MARSHAL", length_mm=390, score=0.99),
            _v2_source_point(article="190", brand="Broen", length_mm=190, score=0.8),
            _v2_source_point(article="180", brand="MARSHAL", length_mm=180, score=0.1),
        ]
        engine = SearchEngine(
            embedder=FakeEmbedder(calls=[]),
            client=V2QdrantClient(points),
            attribute_extractor=StaticAttributeExtractor(raw_brand="MARSHAL", length_mm=180),
        )
        _enable_search_trace(engine)

        response = engine.search_v2("Подбери кран MARSHAL в строительную длину 180", limit=5)
    finally:
        reset_request_id(token)

    assert response.status == "exact_match"
    events = _search_trace_events(caplog)
    stages = [event["stage"] for event in events]

    assert stages == [
        "started",
        "attributes",
        "resolution",
        "constraints",
        "retrieval_query",
        "embedding",
        "qdrant",
        "filtering",
        "ranking",
    ]
    assert {event["request_id"] for event in events} == {"trace-request-1"}
    assert events[1]["attributes"] == {
        "brand": "MARSHAL",
        "article": None,
        "dn": None,
        "pn_bar": None,
        "connection": None,
        "body_material": None,
        "medium": None,
        "control": None,
        "temperature": None,
        "length_mm": 180.0,
        "series": None,
    }

    constraints = next(event for event in events if event["stage"] == "constraints")
    assert constraints["hard"]["brand"] == "MARSHAL"
    assert "length_mm" not in constraints["hard"]
    assert constraints["soft"] == {"length_mm": 180.0}

    qdrant = next(event for event in events if event["stage"] == "qdrant")
    assert qdrant["points_count"] == 3
    assert qdrant["candidate_limit"] == 300
    assert len(qdrant["top_candidates"]) == 3

    filtering = next(event for event in events if event["stage"] == "filtering")
    assert filtering["before"] == 3
    assert filtering["after"] == 2
    assert filtering["rejected"] == 1

    ranking = next(event for event in events if event["stage"] == "ranking")
    assert ranking["soft_length_requested"] == 180.0
    assert ranking["results_count"] == 2
    assert ranking["top_results"] == [
        {"article": "180", "length_mm": 180.0, "length_delta": 0.0},
        {"article": "390", "length_mm": 390.0, "length_delta": 210.0},
    ]

    serialized_events = json.dumps(events, ensure_ascii=False)
    assert '"vector"' not in serialized_events
    assert "[0.1, 0.2, 0.3]" not in serialized_events


def test_search_v2_trace_logs_exact_article(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")
    token = set_request_id("trace-exact")
    try:
        point = _v2_source_point(
            article="ART-1",
            brand="Stout",
            dn=20,
            connection=None,
            length_mm=390,
        )
        engine = SearchEngine(
            embedder=FakeEmbedder(calls=[]),
            client=V2QdrantClient([]),
            attribute_extractor=StaticAttributeExtractor(
                raw_brand="Stout",
                article="ART-1",
                length_mm=180,
            ),
            query_resolver=ExactArticleResolver(dict(point.payload), brand="Stout"),
        )
        _enable_search_trace(engine)

        response = engine.search_v2("ART-1", limit=5)
    finally:
        reset_request_id(token)

    assert response.status == "exact_match"
    events = _search_trace_events(caplog)
    stages = [event["stage"] for event in events]
    assert stages == [
        "started",
        "attributes",
        "article_detected",
        "article_lookup",
        "article_dedup",
        "resolution",
        "constraints",
        "article_resolved",
    ]
    article_detected = next(event for event in events if event["stage"] == "article_detected")
    assert article_detected["article"] == "ART-1"
    assert article_detected["explicit_brand"] is None
    assert article_detected["explicit_dn"] is None
    article_resolved = next(event for event in events if event["stage"] == "article_resolved")
    assert article_resolved["request_id"] == "trace-exact"
    assert article_resolved["source_article"] == "ART-1"
    assert article_resolved["source_brand"] == "Stout"
    assert article_resolved["article_match_type"] == "exact"
    assert article_resolved["search_mode"] == "article_exact"


def test_search_v2_article_only_query_ignores_inferred_hard_constraints() -> None:
    source_product = {
        "article": "ART-1",
        "article_norm": "art1",
        "brand": "Stout",
        "dn": 15.0,
        "pn_bar": 40.0,
        "connection": "flanged",
        "length_mm": 390.0,
        "ld_candidates": [
            _ld_candidate(
                "LD-1",
                "ld1",
                name="LD Stout DN15 PN40",
                dn=15,
                pn_bar=40,
                connection="flanged",
                medium="liquid",
                control="manual",
                url="https://example.invalid/ld-1",
                price=1000,
            )
        ],
    }
    resolver = ExactArticleResolver(source_product, brand="Stout")
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient([]),
        attribute_extractor=StaticAttributeExtractor(
            article="ART-1",
            dn=999,
            pn_bar=999,
            connection="сварное",
        ),
        query_resolver=resolver,
    )

    response = engine.search_v2("Нужен аналог ART-1", limit=5)

    assert resolver.calls[0]["dn"] is None
    assert resolver.calls[0]["pn_bar"] is None
    assert resolver.calls[0]["connection"] is None
    assert response.status == "exact_match"
    assert len(response.results) == 1


def test_search_v2_article_query_respects_explicit_hard_constraints() -> None:
    source_product = {
        "article": "ART-1",
        "article_norm": "art1",
        "brand": "Stout",
        "dn": 15.0,
        "pn_bar": 40.0,
        "connection": "flanged",
        "ld_candidates": [],
    }
    resolver = ExactArticleResolver(source_product, brand="Stout")
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient([]),
        attribute_extractor=StaticAttributeExtractor(
            article="ART-1",
            dn=999,
            pn_bar=999,
            connection="сварное",
        ),
        query_resolver=resolver,
    )

    engine.search_v2("Нужен аналог ART-1 DN100 PN16 фланцевое", limit=5)

    assert resolver.calls[0]["dn"] == 100.0
    assert resolver.calls[0]["pn_bar"] == 16.0
    assert resolver.calls[0]["connection"] == "фланцевое"


def test_search_v2_trace_logs_failure_without_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")
    token = set_request_id("trace-failure")
    try:
        engine = SearchEngine(
            embedder=FailingEmbedder(calls=[]),
            client=V2QdrantClient([]),
            attribute_extractor=StaticAttributeExtractor(raw_brand="MARSHAL"),
        )
        _enable_search_trace(engine)

        with pytest.raises(EmbeddingTimeoutError):
            engine.search_v2("MARSHAL", limit=5)
    finally:
        reset_request_id(token)

    failed = next(event for event in _search_trace_events(caplog) if event["stage"] == "failed")
    assert failed["request_id"] == "trace-failure"
    assert failed["error_type"] == "EmbeddingTimeoutError"
    assert failed["last_completed_stage"] == "retrieval_query"
    assert "timed out" not in json.dumps(failed)


def test_search_v2_trace_respects_feature_flag_but_keeps_summary_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rag_steel.observability")
    token = set_request_id("trace-disabled")
    try:
        engine = SearchEngine(
            embedder=FakeEmbedder(calls=[]),
            client=V2QdrantClient([_v2_source_point(brand="MARSHAL")]),
            attribute_extractor=StaticAttributeExtractor(raw_brand="MARSHAL"),
        )

        response = engine.search_v2("MARSHAL", limit=5)
    finally:
        reset_request_id(token)

    assert response.status == "exact_match"
    events = _logged_events(caplog)
    assert [event for event in events if event["event"] == "search_trace"] == []
    completed = [event for event in events if event["event"] == "search_completed"]
    assert len(completed) == 1
    assert completed[0]["request_id"] == "trace-disabled"
    assert completed[0]["result_status"] == "exact_match"


def test_length_reranking_orders_missing_length_last_without_rejecting() -> None:
    constraints = QueryConstraints(length_mm=180)
    points = [
        _v2_source_point(article="missing", length_mm=None, score=0.99),
        _v2_source_point(article="390", length_mm=390, score=0.8),
        _v2_source_point(article="200", length_mm=200, score=0.7),
        _v2_source_point(article="185", length_mm=185, score=0.6),
        _v2_source_point(article="180", length_mm=180, score=0.1),
    ]
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient(points),
        attribute_extractor=FakeAttributeExtractor(),
    )

    matches = engine._collect_competitor_matches(points, constraints, match_type="exact_match")

    assert [match.competitor.article for match in matches] == [
        "180",
        "185",
        "200",
        "390",
        "missing",
    ]
    assert matches[-1].differences == {"length_mm": {"requested": 180.0, "actual": None}}


def test_search_v2_preserves_score_order_when_length_is_not_requested() -> None:
    constraints = QueryConstraints()
    points = [
        _v2_source_point(article="low", length_mm=180, score=0.1),
        _v2_source_point(article="high", length_mm=390, score=0.9),
    ]
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient(points),
        attribute_extractor=FakeAttributeExtractor(),
    )

    matches = engine._collect_competitor_matches(points, constraints, match_type="exact_match")

    assert [match.competitor.article for match in matches] == ["high", "low"]


@pytest.mark.parametrize(
    ("query", "attributes", "source_kwargs", "expected_status"),
    [
        ("ART-1 латунный", {"body_material": "латунь"}, {"body_material": "латунь"}, "exact_match"),
        ("ART-1 латунный", {"body_material": "латунь"}, {"body_material": "сталь 20"}, "not_found"),
        ("ART-1 series 3", {"series": "3"}, {"name": "Stout series 30 DN20"}, "not_found"),
    ],
)
def test_search_v2_exact_article_validates_hard_attributes(
    query: str,
    attributes: dict[str, object],
    source_kwargs: dict[str, object],
    expected_status: str,
) -> None:
    point = _v2_source_point(
        article="ART-1",
        brand="Stout",
        dn=20,
        connection=None,
        **source_kwargs,
    )
    source_product = dict(point.payload)
    fake_client = V2QdrantClient([])
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=fake_client,
        attribute_extractor=StaticAttributeExtractor(
            raw_brand="Stout",
            article="ART-1",
            **attributes,
        ),
        query_resolver=ExactArticleResolver(source_product, brand="Stout"),
    )

    response = engine.search_v2(query, limit=5)

    assert response.status == expected_status
    assert len(response.results) == (1 if expected_status == "exact_match" else 0)
    assert fake_client.query_calls == []


def test_search_v2_exact_article_keeps_length_soft_and_reports_difference() -> None:
    point = _v2_source_point(
        article="ART-1",
        brand="Stout",
        dn=20,
        connection=None,
        length_mm=390,
    )
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient([]),
        attribute_extractor=StaticAttributeExtractor(
            raw_brand="Stout",
            article="ART-1",
            length_mm=180,
        ),
        query_resolver=ExactArticleResolver(dict(point.payload), brand="Stout"),
    )

    response = engine.search_v2("ART-1", limit=5)

    assert response.status == "exact_match"
    assert len(response.results) == 1
    assert response.results[0].differences == {
        "length_mm": {
            "requested": 180.0,
            "actual": 390.0,
            "delta": 210.0,
        }
    }


def test_search_v2_combines_hard_attributes_and_soft_length() -> None:
    points = [
        _v2_source_point(
            article="A",
            brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="фланцевое",
            body_material="сталь 20",
            control="электропривод",
            length_mm=180,
            score=0.2,
        ),
        _v2_source_point(
            article="B",
            brand="PALUR",
            dn=500,
            pn_bar=25,
            connection="фланцевое",
            body_material="сталь 20",
            control="электропривод",
            length_mm=185,
            score=0.9,
        ),
        _v2_source_point(
            article="C",
            brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="фланцевое",
            body_material="сталь 20",
            control="электропривод",
            length_mm=390,
            score=0.8,
        ),
        _v2_source_point(
            article="D",
            brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="сварное",
            body_material="сталь 20",
            control="электропривод",
            length_mm=180,
        ),
        _v2_source_point(
            article="E",
            brand="PALUR",
            dn=500,
            pn_bar=10,
            connection="фланцевое",
            body_material="сталь 20",
            control="электропривод",
            length_mm=180,
        ),
        _v2_source_point(
            article="F",
            brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="фланцевое",
            body_material="09Г2С",
            control="электропривод",
            length_mm=180,
        ),
    ]
    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=V2QdrantClient(points),
        attribute_extractor=StaticAttributeExtractor(
            raw_brand="PALUR",
            dn=500,
            pn_bar=16,
            connection="фланцевое",
            body_material="сталь 20",
            control="электропривод",
            length_mm=180,
        ),
    )

    response = engine.search_v2(
        "PALUR DN500 PN16 фланцевый сталь 20 электропривод 180 мм",
        limit=10,
    )

    assert [result.competitor.article for result in response.results] == ["A", "B", "C"]


def test_search_v2_short_circuits_without_brand() -> None:
    fake_embedder = FakeEmbedder(calls=[])

    class RecordingQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=[])

    fake_client = RecordingQdrantClient()
    engine = SearchEngine(
        embedder=fake_embedder,
        client=fake_client,
        attribute_extractor=FakeAttributeExtractor(),
    )

    response = engine.search_v2("шаровый кран ду50 ру16", limit=5)

    assert response.status == "cannot_process"
    assert response.reason == {
        "code": "COMPETITOR_BRAND_REQUIRED",
        "message": SEARCH_FAILURE_MESSAGE,
        "retryable": False,
    }
    assert fake_embedder.calls == []
    assert fake_client.query_calls == []


@pytest.mark.parametrize("query", ["Temper DN999", "Temper DN175"])
def test_search_v2_short_circuits_on_unresolved_hard_constraints(query: str) -> None:
    fake_embedder = FakeEmbedder(calls=[])

    class RecordingQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=[])

    fake_client = RecordingQdrantClient()
    engine = SearchEngine(
        embedder=fake_embedder,
        client=fake_client,
        attribute_extractor=DroppedHardConstraintExtractor(),
    )

    response = engine.search_v2(query, limit=5)

    assert response.status == "cannot_process"
    assert response.resolution_mode == "hard_constraint_unresolved"
    assert response.reason == {
        "code": "HARD_CONSTRAINT_UNRESOLVED",
        "message": SEARCH_FAILURE_MESSAGE,
        "retryable": False,
    }
    assert response.requested["brand"] == "Temper"
    assert response.requested["dn"] is None
    assert response.requested["pn_bar"] is None
    assert fake_embedder.calls == []
    assert fake_client.query_calls == []


def test_search_v2_uses_exact_brand_fallback_after_deepseek() -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = EmptyQdrantClient()
    fake_resolver = BrandFallbackResolver()
    engine = SearchEngine(
        embedder=fake_embedder,
        client=fake_client,
        attribute_extractor=NoBrandFallbackExtractor(),
        query_resolver=fake_resolver,
        brand_detector=lambda query: "Temper" if "Temper" in query else None,
    )

    response = engine.search_v2("Нужен аналог крана Temper Ду 20 фланцевого Ру 40", limit=5)

    assert response.status == "not_found"
    assert fake_resolver.calls[0]["raw_brand"] is None
    assert fake_resolver.calls[1]["raw_brand"] == "Temper"
    assert response.requested["brand"] == "Temper"
    assert response.requested["dn"] == 20
    assert response.requested["pn_bar"] == 40
    assert fake_client.query_calls
    assert fake_embedder.calls


def test_search_v2_preserves_raw_article_on_article_not_found() -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = EmptyQdrantClient()
    engine = SearchEngine(
        embedder=fake_embedder,
        client=fake_client,
        attribute_extractor=RawArticleNotFoundExtractor(),
        query_resolver=ArticleNotFoundResolver(),
    )

    response = engine.search_v2("Подбери аналог для крана 107-5529ШШ", limit=5)

    assert response.status == "not_found"
    assert response.reason == {
        "code": "ARTICLE_NOT_FOUND",
        "message": SEARCH_FAILURE_MESSAGE,
        "retryable": False,
    }
    assert response.requested["article"] == "107-5529ШШ"
    assert response.requested.get("resolved_article") is None
    assert fake_client.query_calls == []
    assert fake_embedder.calls == []


def test_readiness_reports_missing_deepseek_configuration_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_settings = Settings(
        embedding_model="fake",
        embedding_dimension=3,
        openai_api_key="test-key",
        openai_base_url="https://example.invalid/v1",
        openai_timeout_seconds=12.0,
        deepseek_api_key="",
        deepseek_base_url="https://example.invalid/v1",
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=12.0,
        dense_batch_size=4,
        max_concurrent_searches=8,
        qdrant_timeout_seconds=5.0,
        upstream_max_attempts=2,
        upstream_retry_base_delay_seconds=0.25,
        qdrant_url="http://localhost:6333",
        qdrant_collection_alias="steel_products_active",
        qdrant_dense_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
        source_candidate_limit=10,
        dense_score_threshold=None,
        bm25_score_threshold=None,
        result_limit_default=20,
        result_limit_max=100,
        search_trace_enabled=False,
    )
    monkeypatch.setattr(search_engine_mod, "get_settings", lambda: fake_settings)

    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=MissingAliasQdrantClient(),
        attribute_extractor=SimpleNamespace(extract=lambda query: None),
    )

    ready, payload = engine.readiness_status()

    assert ready is False
    assert payload["reason"] == "DEEPSEEK_CONFIGURATION_MISSING"
    assert payload["details"]["deepseek_configured"] is False
    assert payload["details"]["deepseek_model"] == "deepseek-v4-flash"
    assert payload["details"]["resolved_collection_name"] is None


def test_search_v2_applies_query_filter_before_qdrant_retrieval() -> None:
    class RecordingQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="doc-1",
                        score=0.91,
                        payload={
                            "article": "1184399",
                            "article_norm": "1184399",
                            "name": "Temper DN80 PN16",
                            "brand": "Temper",
                            "dn": 80,
                            "pn_bar": 16,
                            "connection": "flanged",
                            "body_material": "сталь 09Г2С",
                            "ld_candidates": [_v2_ld_candidate()],
                        },
                    )
                ]
            )

    engine = SearchEngine(
        embedder=FakeEmbedder(calls=[]),
        client=RecordingQdrantClient(),
        attribute_extractor=FakeAttributeExtractor(),
    )

    response = engine.search_v2("Temper DN80 PN16", limit=5)

    assert response.status == "exact_match"
    assert response.results
    assert engine._client is not None
    assert engine._client.query_calls[0]["query_filter"] is not None
    assert len(engine._client.query_calls[0]["prefetch"]) == 2


def test_search_deduplicates_ld_candidates_and_builds_evidence() -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    response = engine.search("Temper 1184399 DN80 PN16", limit=5)

    assert isinstance(response, SearchResponse)
    assert response.query == "Temper 1184399 DN80 PN16"
    assert response.count == 3
    assert [result.rank for result in response.results] == [1, 2, 3]
    assert len({result.product["article_norm"] for result in response.results}) == 3
    assert response.results[0].product["article_norm"] == "11100800162muld000003000"
    assert response.results[1].product["article_norm"] == "11100800162muld000005000"
    assert response.results[2].product["article_norm"] == "11100800162muld000004000"
    assert len(fake_embedder.calls) == 1
    assert fake_embedder.calls[0]["texts"] == ["Temper 1184399 DN80 PN16"]
    assert fake_client.query_calls[0]["prefetch"][0].score_threshold is None
    assert fake_client.query_calls[0]["prefetch"][1].score_threshold is None


def test_search_preserves_qdrant_source_order() -> None:
    class OrderedQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="A",
                        score=0.9,
                        payload={
                            "article_norm": "a",
                            "ld_candidates": [
                                _ld_candidate(
                                    "A-LD",
                                    "a-ld",
                                    name="LD A",
                                    dn=80,
                                    pn_bar=16,
                                    connection="flanged",
                                    medium="liquid",
                                    control="manual",
                                    url="https://example.invalid/a",
                                    price=1,
                                )
                            ],
                        },
                    ),
                    SimpleNamespace(
                        id="B",
                        score=0.9,
                        payload={
                            "article_norm": "b",
                            "ld_candidates": [
                                _ld_candidate(
                                    "B-LD",
                                    "b-ld",
                                    name="LD B",
                                    dn=80,
                                    pn_bar=16,
                                    connection="flanged",
                                    medium="liquid",
                                    control="manual",
                                    url="https://example.invalid/b",
                                    price=2,
                                )
                            ],
                        },
                    ),
                    SimpleNamespace(
                        id="C",
                        score=0.8,
                        payload={
                            "article_norm": "c",
                            "ld_candidates": [
                                _ld_candidate(
                                    "C-LD",
                                    "c-ld",
                                    name="LD C",
                                    dn=80,
                                    pn_bar=16,
                                    connection="flanged",
                                    medium="liquid",
                                    control="manual",
                                    url="https://example.invalid/c",
                                    price=3,
                                )
                            ],
                        },
                    ),
                ]
            )

    fake_embedder = FakeEmbedder(calls=[])
    fake_client = OrderedQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    response = engine.search("Temper DN80 PN16", limit=10)

    assert [result.product["article"] for result in response.results] == ["A-LD", "B-LD", "C-LD"]
    assert [result.score for result in response.results] == [0.9, 0.9, 0.8]


def test_search_sorts_deduped_ld_candidates_by_best_source_score_before_limit() -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    response = engine.search("Temper 1184399 DN80 PN16", limit=2)

    assert [result.product["article_norm"] for result in response.results] == [
        "11100800162muld000003000",
        "11100800162muld000005000",
    ]
    assert [result.score for result in response.results] == [0.97, 0.97]


def test_search_leaves_raw_query_untouched_for_embedder() -> None:
    original_query = "  TEMPER   Du80/Du16?  "
    fake_embedder = FakeEmbedder(calls=[])

    class RecordingQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=[])

    fake_client = RecordingQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    engine.search(original_query, limit=1)

    assert fake_embedder.calls[0]["texts"] == [original_query]
    assert fake_client.query_calls[0]["prefetch"][1].query.text == original_query


def test_search_passes_independent_thresholds_into_qdrant_prefetch() -> None:
    class ThresholdQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="doc-1",
                        score=0.02,
                        payload={
                            "article": "1184399",
                            "article_norm": "1184399",
                            "name": "Temper DN80 PN16",
                            "ld_candidates": [
                                _ld_candidate(
                                    "11100800162MULD000003000",
                                    "11100800162muld000003000",
                                    name="LD Temper DN80 PN16",
                                    dn=80,
                                    pn_bar=16,
                                    connection="flanged",
                                    medium="liquid",
                                    control="manual",
                                    url="https://example.invalid/ld-a",
                                    price=12130,
                                )
                            ],
                        },
                    )
                ]
            )

    fake_embedder = FakeEmbedder(calls=[])
    fake_client = ThresholdQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)
    engine.settings = SimpleNamespace(
        qdrant_dense_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
        embedding_dimension=3,
        dense_score_threshold=0.75,
        bm25_score_threshold=4.0,
        upstream_max_attempts=2,
        upstream_retry_base_delay_seconds=0.25,
    )

    response = engine.search("Temper DN80 PN16", limit=1)

    query_call = fake_client.query_calls[0]
    assert query_call["prefetch"][0].score_threshold == 0.75
    assert query_call["prefetch"][1].score_threshold == 4.0
    assert response.count == 1
    assert response.results[0].score == pytest.approx(0.02)


def test_search_retries_transient_qdrant_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_embedder = FakeEmbedder(calls=[])
    sleep_calls: list[float] = []

    class FlakyQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []
            self.calls = 0

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("service unavailable")
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="doc-1",
                        score=0.91,
                        payload={
                            "article": "1184399",
                            "article_norm": "1184399",
                            "name": "Temper DN80 PN16",
                            "ld_candidates": [
                                _ld_candidate(
                                    "11100800162MULD000003000",
                                    "11100800162muld000003000",
                                    name="LD Temper DN80 PN16",
                                    dn=80,
                                    pn_bar=16,
                                    connection="flanged",
                                    medium="liquid",
                                    control="manual",
                                    url="https://example.invalid/ld-a",
                                    price=12130,
                                )
                            ],
                        },
                    )
                ]
            )

    monkeypatch.setattr(search_engine_mod, "sleep", lambda seconds: sleep_calls.append(seconds))

    engine = SearchEngine(embedder=fake_embedder, client=FlakyQdrantClient())
    engine.settings = SimpleNamespace(
        qdrant_dense_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
        dense_score_threshold=None,
        bm25_score_threshold=None,
        upstream_max_attempts=2,
        upstream_retry_base_delay_seconds=0.25,
    )
    response = engine.search("Temper DN80 PN16", limit=1)

    assert response.count == 1
    assert response.results[0].product["article_norm"] == "11100800162muld000003000"
    assert sleep_calls == [0.25]


def test_search_raises_timeout_after_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_embedder = FakeEmbedder(calls=[])
    sleep_calls: list[float] = []

    class TimeoutQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []
            self.calls = 0

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            self.calls += 1
            raise RuntimeError("timed out")

    monkeypatch.setattr(search_engine_mod, "sleep", lambda seconds: sleep_calls.append(seconds))

    engine = SearchEngine(embedder=fake_embedder, client=TimeoutQdrantClient())
    engine.settings = SimpleNamespace(
        qdrant_dense_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
        dense_score_threshold=None,
        bm25_score_threshold=None,
        upstream_max_attempts=2,
        upstream_retry_base_delay_seconds=0.25,
    )

    with pytest.raises(SearchBackendTimeoutError):
        engine.search("Temper DN80 PN16", limit=1)

    assert sleep_calls == []


def test_search_does_not_fallback_to_collection_from_build_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "index_build_baai-bge-m3.json").write_text(
        (
            "{"
            '"embedding_model":"BAAI/bge-m3",'
            '"collection_alias":"steel_products_active",'
            '"collection_name":"steel_products_baai-bge-m3_20260806T132928Z"'
            "}"
        ),
        encoding="utf-8",
    )

    fake_embedder = FakeEmbedder(calls=[])
    fake_client = MissingAliasQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    with pytest.raises(UnexpectedResponse):
        engine.search("Temper DN80 PN16", limit=1)

    assert [call["collection_name"] for call in fake_client.query_calls] == [
        "steel_products_active"
    ]


def test_readiness_reports_missing_alias_as_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_STEEL_DISABLE_DOTENV", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "index_build_baai-bge-m3.json").write_text(
        (
            "{"
            '"embedding_model":"BAAI/bge-m3",'
            '"collection_alias":"steel_products_active",'
            '"collection_name":"steel_products_baai-bge-m3_20260806T132928Z"'
            "}"
        ),
        encoding="utf-8",
    )

    fake_client = MissingAliasQdrantClient()
    engine = SearchEngine(embedder=FakeEmbedder(calls=[]), client=fake_client)

    ready, payload = engine.readiness_status()

    assert ready is False
    assert payload["reason"] == "QDRANT_COLLECTION_MISSING"
    assert payload["details"]["deepseek_configured"] is True
    assert payload["collection_alias"] == "steel_products_active"
    assert payload["resolved_collection_name"] is None
    assert payload["details"]["resolved_collection_name"] is None
    assert fake_client.get_collection_calls == ["steel_products_active"]


def test_readiness_resolves_alias_to_physical_collection_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_STEEL_DISABLE_DOTENV", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "index_build_baai-bge-m3.json").write_text(
        (
            "{"
            '"embedding_model":"BAAI/bge-m3",'
            '"collection_alias":"steel_products_active",'
            '"collection_name":"steel_products_20260817T010203Z"'
            "}"
        ),
        encoding="utf-8",
    )

    fake_client = AliasedQdrantClient()
    engine = SearchEngine(embedder=FakeEmbedder(calls=[]), client=fake_client)

    ready, payload = engine.readiness_status()

    assert ready is True
    assert payload["resolved_collection_name"] == "steel_products_20260817T010203Z"
    assert payload["details"]["resolved_collection_name"] == "steel_products_20260817T010203Z"
    assert payload["qdrant"]["resolved_collection"] == "steel_products_20260817T010203Z"
    assert fake_client.get_collection_calls == ["steel_products_active"]


def test_search_prefers_alias_after_readiness_resolves_physical_name() -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = HotAliasSwitchQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    ready, payload = engine.readiness_status()
    assert ready is True
    assert payload["resolved_collection_name"] == "collection_A"

    fake_client._alias_target = "collection_B"
    response = engine.search("Temper DN80 PN16", limit=20)

    assert response.count == 1
    assert fake_client.query_calls[0] == "steel_products_active"
    assert "collection_A" not in fake_client.query_calls


@pytest.mark.parametrize(
    ("query", "expected_count"),
    [
        ("1184399", 7),
        ("A0486", 7),
        ("A-0486", 7),
        ("KSH.P.P.015.40-01", 7),
        ("kshpp0154001", 7),
        ("Temper DN80 PN16", 20),
        ("Temper 1184399 DN80 PN16", 20),
        ("Broen DN80 PN16", 20),
        ("flanged valve DN50 PN40", 20),
        ("nonexistent item", 0),
        ("", 0),
    ],
)
def test_search_regressions_cover_expected_queries(query: str, expected_count: int) -> None:
    fake_embedder = FakeEmbedder(calls=[])
    fake_client = RegressionQdrantClient()
    engine = SearchEngine(embedder=fake_embedder, client=fake_client)

    response = engine.search(query, limit=20)

    assert response.count == expected_count
    assert len(response.results) == expected_count
    assert len({result.product["article_norm"] for result in response.results}) == expected_count


@pytest.mark.parametrize(
    ("metadata", "actual_dimension", "expected_compatible", "expected_reason"),
    [
        (
            {
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            3,
            True,
            None,
        ),
        (
            {
                "point_count": 16016,
            },
            3,
            False,
            "INDEX_METADATA_INCOMPLETE",
        ),
        (
            {
                "schema_version": "v1",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            3,
            False,
            "SCHEMA_VERSION_MISMATCH",
        ),
        (
            {
                "schema_version": "v2",
                "index_schema_version": 1,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            3,
            False,
            "INDEX_SCHEMA_VERSION_MISMATCH",
        ),
        (
            {
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            4,
            False,
            "VECTOR_DIMENSION_MISMATCH",
        ),
        (
            {
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": SUPPORTED_INDEX_FORMAT_VERSION,
                "embedding_model": "other",
                "embedding_dimension": 3,
            },
            3,
            False,
            "EMBEDDING_MODEL_MISMATCH",
        ),
        (
            {
                "schema_version": "v2",
                "index_schema_version": 2,
                "index_format_version": 99,
                "embedding_model": "fake",
                "embedding_dimension": 3,
            },
            3,
            False,
            "INDEX_FORMAT_UNSUPPORTED",
        ),
        (None, 3, True, None),
    ],
)
def test_check_index_compatibility_covers_expected_cases(
    metadata: dict[str, object] | None,
    actual_dimension: int,
    expected_compatible: bool,
    expected_reason: str | None,
) -> None:
    result = check_index_compatibility(
        metadata=metadata,
        actual_dimension=actual_dimension,
        settings=SimpleNamespace(embedding_model="fake", embedding_dimension=3),
    )

    assert result["compatible"] is expected_compatible
    assert result["reason"] == expected_reason
    if metadata is None:
        assert "INDEX_METADATA_MISSING" in result["warnings"]
