from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

import rag_steel.search_engine as search_engine_mod
from rag_steel.normalization import normalize_text
from rag_steel.search_engine import SearchEngine, SearchResponse
from rag_steel.runtime import SearchBackendTimeoutError


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
        if collection_name == "steel_products_active":
            raise self._missing_collection(collection_name)
        return SimpleNamespace(
            metadata={
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

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        if kwargs["collection_name"] == "steel_products_active":
            raise self._missing_collection("steel_products_active")
        return super().query_points(**kwargs)


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

    with pytest.raises(SearchBackendTimeoutError):
        engine.search("Temper DN80 PN16", limit=1)

    assert sleep_calls == [0.25]


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
    assert payload["collection_alias"] == "steel_products_active"
    assert payload["resolved_collection_name"] is None
    assert payload["details"]["resolved_collection_name"] is None
    assert fake_client.get_collection_calls == ["steel_products_active"]


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
