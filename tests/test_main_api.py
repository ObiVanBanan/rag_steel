from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import main
from rag_steel.runtime import SearchBackendTimeoutError, SearchConcurrencyGate
from rag_steel.search_engine import CompetitorMatch, CompetitorProduct, SearchResponse, SearchResult


class FakeClient:
    def get_collection(self, **_: object) -> object:
        return SimpleNamespace(
            metadata={
                "index_schema_version": 2,
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": "",
                "embedding_dimension": 1024,
            },
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=1024)})
            ),
        )

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
        return SearchResponse(
            query=query,
            count=1,
            results=[
                SearchResult(
                    rank=1,
                    score=0.97,
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
                    source_evidence=[
                        {"source_article": "1184399", "source_name": "Temper DN80 PN16"}
                    ],
                )
            ],
            timing_ms={"embedding": 0.2, "qdrant": 0.3, "ranking": 0.4},
        )

    def search_v2(self, query: str, limit: int = 20, **_: object) -> object:
        self.search_calls.append({"query": query, "limit": limit, "kind": "v2"})
        return SimpleNamespace(
            request_id="11111111-1111-1111-1111-111111111111",
            query=query,
            status="exact_match",
            requested={"brand": "Temper", "dn": 80, "pn_bar": 16},
            results=[
                CompetitorMatch(
                    match_type="exact_match",
                    differences={},
                    competitor=CompetitorProduct(
                        article="1184399",
                        name="Temper DN80 PN16",
                        brand="Temper",
                        dn=80,
                        pn_bar=16,
                        connection="flanged",
                        body_material="сталь 09г2с",
                    ),
                    ld_articles=["11100800162MULD000003000"],
                )
            ],
            timing_ms={"embedding": 0.2, "qdrant": 0.3, "ranking": 0.4},
        )

    def _get_client(self) -> FakeClient:
        return self._client

    def _get_model(self) -> object:
        return self._model

    def readiness_status(self) -> tuple[bool, dict[str, object]]:
        return (
            True,
            {
                "status": "ok",
                "collection_alias": self.collection_alias,
                "point_count": 7,
                "details": {
                    "runtime_model": "BAAI/bge-m3",
                    "index_model": "BAAI/bge-m3",
                },
            },
        )


class VariableResponseEngine:
    def __init__(
        self,
        *,
        result_count: int,
        duplicate_ld: bool = False,
        qdrant_error: Exception | None = None,
    ) -> None:
        self.model_name = "fake-model"
        self.collection_alias = "steel_products_active"
        self.result_count = result_count
        self.duplicate_ld = duplicate_ld
        self.qdrant_error = qdrant_error
        self.search_calls: list[dict[str, object]] = []
        self._client = FakeClient()
        self._model = object()

    def search(self, query: str, limit: int = 20, **_: object) -> SearchResponse:
        self.search_calls.append({"query": query, "limit": limit})
        results: list[SearchResult] = []
        for index in range(min(self.result_count, limit)):
            article = (
                "11100800162MULD000003000" if self.duplicate_ld else f"11100800162MULD{index:09d}"
            )
            results.append(
                SearchResult(
                    rank=index + 1,
                    score=0.9,
                    product={
                        "article": article,
                        "article_norm": article.lower(),
                        "name": f"LD #{index + 1}",
                        "url": f"https://example.invalid/{index + 1}",
                    },
                    source_evidence=[],
                )
            )
        return SearchResponse(
            query=query,
            count=len(results),
            results=results,
            timing_ms={"embedding": 0.2, "qdrant": 0.3, "ranking": 0.4},
        )

    def _get_client(self) -> FakeClient:
        if self.qdrant_error is not None:
            raise self.qdrant_error
        return self._client

    def _get_model(self) -> object:
        if self.qdrant_error is not None:
            return self._model
        return self._model

    def readiness_status(self) -> tuple[bool, dict[str, object]]:
        if self.qdrant_error is not None:
            raise self.qdrant_error
        return (
            True,
            {
                "status": "ok",
                "collection_alias": self.collection_alias,
                "point_count": 7,
                "details": {
                    "runtime_model": "BAAI/bge-m3",
                    "index_model": "BAAI/bge-m3",
                },
            },
        )


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
        assert body["results"][0]["score"] == 0.97
        assert body["results"][0]["product"]["article"] == "11100800162MULD000003000"
        assert body["results"][0]["source_evidence"] == [
            {"source_article": "1184399", "source_name": "Temper DN80 PN16"}
        ]
        assert body["timing_ms"]["ranking"] == 0.4
        assert client.fake_engine.search_calls == [{"query": "Temper DN80 PN16", "limit": 20}]


def test_v2_search_returns_exact_match_envelope() -> None:
    with _make_client() as client:
        response = client.post(
            "/v2/search",
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
        assert body["status"] == "exact_match"
        assert body["requested"] == {"brand": "Temper", "dn": 80, "pn_bar": 16}
        assert body["results"][0]["match_type"] == "exact_match"
        assert body["results"][0]["differences"] == {}
        assert body["results"][0]["competitor"]["article"] == "1184399"
        assert body["results"][0]["ld_articles"] == ["11100800162MULD000003000"]
        assert client.fake_engine.search_calls[-1] == {
            "query": "Temper DN80 PN16",
            "limit": 20,
            "kind": "v2",
        }


def test_v2_search_returns_not_found_without_fallback() -> None:
    class NotFoundEngine(FakeEngine):
        def search_v2(self, query: str, limit: int = 20, **_: object) -> object:
            return SimpleNamespace(
                request_id="22222222-2222-2222-2222-222222222222",
                query=query,
                status="not_found",
                requested={"brand": "Temper", "dn": 80, "pn_bar": 25},
                results=[],
                timing_ms={"embedding": 0.2, "qdrant": 0.3, "ranking": 0.4},
            )

    engine = NotFoundEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: engine
    with TestClient(main.app) as client:
        response = client.post(
            "/v2/search",
            json={
                "query": "Temper DN80 PN25",
                "limit": 20,
                "include_debug": False,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_found"
        assert body["results"] == []
    main.app.dependency_overrides.clear()


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
        assert response.json()["debug"] == {"pipeline": "raw_query_dense_bm25_rrf"}


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


def test_health_endpoints_ignore_search_gate() -> None:
    gate = SearchConcurrencyGate(1)
    assert gate.try_acquire()

    fake_engine = FakeEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: fake_engine
    main.app.dependency_overrides[main.get_search_gate] = lambda: gate

    try:
        with TestClient(main.app) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")

            assert live.status_code == 200
            assert live.json() == {"status": "ok"}
            assert ready.status_code == 200
            assert ready.json()["status"] == "ok"
    finally:
        gate.release()
        main.app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"query": "", "limit": 20, "include_debug": False}, "query"),
        ({"query": "x" * 513, "limit": 20, "include_debug": False}, "query"),
        ({"query": "Temper DN80 PN16", "limit": 0, "include_debug": False}, "limit"),
        ({"query": "Temper DN80 PN16", "limit": 101, "include_debug": False}, "limit"),
    ],
)
def test_v1_search_validates_input(payload: dict[str, object], field: str) -> None:
    with _make_client() as client:
        response = client.post("/v1/search", json=payload)

        assert response.status_code == 422
        assert field in response.text


def test_v1_search_returns_busy_when_gate_is_exhausted() -> None:
    gate = SearchConcurrencyGate(1)
    assert gate.try_acquire()

    fake_engine = FakeEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: fake_engine
    main.app.dependency_overrides[main.get_search_gate] = lambda: gate

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/v1/search",
                json={
                    "query": "Temper DN80 PN16",
                    "limit": 20,
                    "include_debug": False,
                },
            )

            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
            assert response.json()["error"] == {
                "code": "SERVICE_BUSY",
                "message": "Search service is temporarily busy",
            }
    finally:
        gate.release()
        main.app.dependency_overrides.clear()


def test_v1_search_maps_backend_timeout_to_gateway_timeout() -> None:
    class TimeoutEngine(FakeEngine):
        def search(self, query: str, limit: int = 20, **_: object) -> SearchResponse:
            raise SearchBackendTimeoutError("Qdrant query timed out")

    engine = TimeoutEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: engine
    with TestClient(main.app) as client:
        response = client.post(
            "/v1/search",
            json={
                "query": "Temper DN80 PN16",
                "limit": 20,
                "include_debug": False,
            },
        )

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "SEARCH_BACKEND_TIMEOUT"
    main.app.dependency_overrides.clear()


def test_health_ready_reports_unavailable_and_missing_collection() -> None:
    class BrokenClient(FakeClient):
        def get_collection(self, **_: object) -> object:
            raise RuntimeError("Qdrant unavailable")

    class MissingCollectionClient(FakeClient):
        def get_collection(self, **_: object) -> object:
            return SimpleNamespace()

        def count(self, **_: object) -> object:
            return SimpleNamespace(count=0)

    class BrokenEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self._client = BrokenClient()

        def readiness_status(self) -> tuple[bool, dict[str, object]]:
            raise RuntimeError("Qdrant unavailable")

    class MissingCollectionEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self._client = MissingCollectionClient()

        def readiness_status(self) -> tuple[bool, dict[str, object]]:
            return (
                False,
                {
                    "status": "not_ready",
                    "reason": "EMPTY_COLLECTION",
                    "details": {"point_count": 0},
                },
            )

    for engine_cls in (BrokenEngine, MissingCollectionEngine):
        engine = engine_cls()
        main.app.dependency_overrides[main.get_engine] = lambda engine=engine: engine
        with TestClient(main.app) as client:
            ready = client.get("/health/ready")
            assert ready.status_code == 503
        main.app.dependency_overrides.clear()


def test_health_ready_reports_embedding_index_mismatch() -> None:
    class MismatchEngine(FakeEngine):
        def readiness_status(self) -> tuple[bool, dict[str, object]]:
            return (
                False,
                {
                    "status": "not_ready",
                    "reason": "EMBEDDING_INDEX_MISMATCH",
                    "details": {
                        "runtime_model": "BAAI/bge-m3",
                        "index_model": "intfloat/multilingual-e5-base",
                    },
                },
            )

    engine = MismatchEngine()
    main.app.dependency_overrides[main.get_engine] = lambda: engine
    with TestClient(main.app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["reason"] == "EMBEDDING_INDEX_MISMATCH"
        assert ready.json()["details"]["runtime_model"] == "BAAI/bge-m3"
        assert ready.json()["details"]["index_model"] == "intfloat/multilingual-e5-base"
    main.app.dependency_overrides.clear()


def test_v1_search_handles_zero_and_short_result_sets() -> None:
    cases = [
        (0, 0),
        (7, 7),
        (20, 20),
    ]

    for result_count, expected_count in cases:
        engine = VariableResponseEngine(result_count=result_count)
        main.app.dependency_overrides[main.get_engine] = lambda engine=engine: engine
        with TestClient(main.app) as client:
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
            assert body["count"] == expected_count
            assert len(body["results"]) == expected_count
        main.app.dependency_overrides.clear()
