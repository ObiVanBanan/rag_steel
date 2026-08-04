from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

import main
from rag_steel.query_processor import QueryProcessor
from rag_steel.search_engine import SearchResponse, SearchResult


class FakeClient:
    def get_collection(self, **_: object) -> object:
        return SimpleNamespace()

    def count(self, **_: object) -> object:
        return SimpleNamespace(count=7)


class FakeEngine:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.collection_alias = "steel_products_active"
        self.search_calls: list[dict[str, object]] = []
        self._client = FakeClient()
        self._model = object()

    def search(self, query: str, limit: int = 20, **_: object) -> SearchResponse:
        self.search_calls.append({"query": query, "limit": limit})
        processed = QueryProcessor().process(query)
        return SearchResponse(
            query=query,
            processed_query=processed,
            count=1,
            results=[
                SearchResult(
                    rank=1,
                    relevance_rating=97.4,
                    product={
                        "article": "11100800162MULD000003000",
                        "name": "LD Temper DN80 PN16",
                        "url": "https://example.invalid/ld-a",
                        "price": 12130,
                        "dn": 80,
                        "pn_bar": 16,
                        "connection": "фланцевое",
                        "medium": "жидкость",
                        "control": "ручное",
                    },
                    match_reasons=["Совпадает DN 80", "Совпадает PN 16"],
                    mismatches=[],
                    source_evidence=[
                        {"source_article": "1184399", "source_name": "Temper DN80 PN16"}
                    ],
                    score_breakdown={
                        "hybrid_score": 0.96,
                        "text_exactness": 1.0,
                        "source_field_score": 1.0,
                        "source_score": 0.97,
                        "ld_field_score": 1.0,
                        "final_score": 0.974,
                    },
                )
            ],
            timing_ms={"normalize": 0.1, "embedding": 0.2, "qdrant": 0.3, "ranking": 0.4},
        )

    def _get_client(self) -> FakeClient:
        return self._client

    def _get_model(self) -> object:
        return self._model


@contextmanager
def _make_client():
    fake_engine = FakeEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: fake_engine
    with TestClient(main.app) as client:
        client.fake_engine = fake_engine  # type: ignore[attr-defined]
        yield client
    main.app.dependency_overrides.clear()


def test_v1_search_returns_unified_api_response() -> None:
    with _make_client() as client:
        response = client.post(
            "/v1/search",
            json={
                "query": "Temper DN80 PN16",
                "limit": 20,
                "include_debug": False,
            },
        )

        assert response.status_code == 200
        body = response.json()
        UUID(body["request_id"])
        assert body["query"] == "Temper DN80 PN16"
        assert body["count"] == 1
        assert "debug" not in body
        assert body["results"][0]["rank"] == 1
        assert body["results"][0]["relevance_rating"] == 97.4
        assert body["results"][0]["product"]["article"] == "11100800162MULD000003000"
        assert body["results"][0]["match_reasons"] == ["Совпадает DN 80", "Совпадает PN 16"]
        assert body["results"][0]["source_evidence"] == [
            {"source_article": "1184399", "source_name": "Temper DN80 PN16"}
        ]
        assert body["timing_ms"]["ranking"] == 0.4
        assert client.fake_engine.search_calls == [{"query": "Temper DN80 PN16", "limit": 20}]


def test_legacy_wrappers_delegate_to_search_engine() -> None:
    with _make_client() as client:
        response = client.post(
            "/search",
            json={
                "query": "Temper DN80 PN16",
                "top_k": 5,
                "use_hybrid": True,
            },
        )

        assert response.status_code == 200
        assert client.fake_engine.search_calls[-1] == {"query": "Temper DN80 PN16", "limit": 5}

        response = client.post(
            "/analogs",
            json={
                "query": "Temper DN80 PN16",
                "limit": 7,
                "include_debug": True,
            },
        )

        assert response.status_code == 200
        assert client.fake_engine.search_calls[-1] == {"query": "Temper DN80 PN16", "limit": 7}
        assert "processed_query" in response.json()["debug"]


def test_health_endpoints_and_removed_compare_models() -> None:
    with _make_client() as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        legacy = client.get("/compare-models?query=Temper")

        assert live.status_code == 200
        assert live.json() == {"status": "ok"}
        assert ready.status_code == 200
        assert ready.json()["status"] == "ok"
        assert ready.json()["point_count"] == 7
        assert legacy.status_code == 404
