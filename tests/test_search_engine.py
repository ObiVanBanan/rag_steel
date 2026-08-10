from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.normalization import normalize_text
from rag_steel.search_engine import SearchEngine, SearchResponse


@dataclass(slots=True)
class FakeModel:
    calls: list[dict[str, object]]

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
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
                        "steel_id": "doc-1",
                        "article": "1184399",
                        "article_norm": "1184399",
                        "article_compact": "1184399",
                        "name": "Temper DN80 PN16",
                        "brand": "Temper",
                        "dn": 80,
                        "pn_bar": 16,
                        "connection": "flanged",
                        "medium": "liquid",
                        "control": "manual",
                        "url": "https://example.invalid/doc-1",
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
                        "steel_id": "doc-2",
                        "article": "a0486",
                        "article_norm": "a0486",
                        "article_compact": "a0486",
                        "name": "Broen DN50 PN10",
                        "brand": "Broen",
                        "dn": 50,
                        "pn_bar": 10,
                        "connection": "threaded",
                        "medium": "gas",
                        "control": "electric",
                        "url": "https://example.invalid/doc-2",
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


class RegressionQdrantClient:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        sparse_query = kwargs["prefetch"][1].query.text  # type: ignore[index]
        normalized = normalize_text(sparse_query) or ""

        if not normalized or "nonexistent" in normalized or "empty query" in normalized:
            return SimpleNamespace(points=[])

        if any(
            token in normalized
            for token in [
                "temper",
                "broen",
                "dn80 pn16",
                "flanged",
            ]
        ):
            result_count = 20
        else:
            result_count = 7

        points = []
        for index in range(result_count):
            candidate_article = f"ld-{index:02d}"
            points.append(
                SimpleNamespace(
                    id=f"source-{index}",
                    score=1.0 - (index * 0.01),
                    payload={
                        "steel_id": f"source-{index}",
                        "article": f"SOURCE-{index}",
                        "article_norm": f"source-{index}",
                        "name": f"Source {index}",
                        "brand": "Temper",
                        "dn": 80 if result_count == 20 else 50,
                        "pn_bar": 16,
                        "connection": "flanged",
                        "medium": "liquid",
                        "control": "manual",
                        "url": f"https://example.invalid/source-{index}",
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
        self.count_calls: list[str] = []

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
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": "",
                "embedding_dimension": 1024,
            },
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=1024)})
            ),
        )

    def count(self, *, collection_name: str, **_: object) -> object:
        self.count_calls.append(collection_name)
        return SimpleNamespace(count=7)

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        collection_name = kwargs["collection_name"]
        if collection_name == "steel_products_active":
            raise self._missing_collection(str(collection_name))
        return super().query_points(**kwargs)


def test_search_deduplicates_ld_candidates_and_builds_evidence() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper 1184399 DN80 PN16", limit=5)

    assert isinstance(response, SearchResponse)
    assert response.query == "Temper 1184399 DN80 PN16"
    assert response.count == 3
    assert [result.rank for result in response.results] == [1, 2, 3]
    assert len({result.product["article_norm"] for result in response.results}) == 3

    top = response.results[0]
    assert top.id == "11100800162muld000003000"
    assert top.product["article"] == "11100800162MULD000003000"
    assert top.product["article_norm"] == "11100800162muld000003000"
    assert top.score == pytest.approx(0.97)
    assert len(top.source_evidence) == 2
    assert [item["source_article"] for item in top.source_evidence] == [
        "1184399",
        "a0486",
    ]
    assert [item["source_rank"] for item in top.source_evidence] == [1, 2]
    assert all(item["source_score"] is not None for item in top.source_evidence)

    second = response.results[1]
    assert second.product["article_norm"] == "11100800162muld000005000"
    assert second.score == pytest.approx(0.97)

    third = response.results[2]
    assert third.product["article_norm"] == "11100800162muld000004000"
    assert third.score == pytest.approx(0.91)

    assert "embedding" in response.timing_ms
    assert "qdrant" in response.timing_ms
    assert "ranking" in response.timing_ms
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["texts"] == ["Temper 1184399 DN80 PN16"]
    assert fake_model.calls[0]["normalize_embeddings"] is True
    assert len(fake_client.query_calls) == 1

    query_call = fake_client.query_calls[0]
    assert query_call["collection_name"] == "steel_products_active"
    assert query_call["limit"] == 300
    assert query_call["with_payload"] is True
    assert len(query_call["prefetch"]) == 2
    assert query_call["prefetch"][0].using == "dense"
    assert query_call["prefetch"][1].using == "sparse"
    assert query_call["prefetch"][0].limit == 300
    assert query_call["prefetch"][1].limit == 300
    assert query_call["prefetch"][1].query.text == "Temper 1184399 DN80 PN16"
    assert query_call["query"].fusion == "rrf"


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

    fake_model = FakeModel(calls=[])
    fake_client = OrderedQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper DN80 PN16", limit=10)

    assert [result.product["article"] for result in response.results] == ["A-LD", "B-LD", "C-LD"]
    assert [result.score for result in response.results] == [0.9, 0.9, 0.8]


def test_search_sorts_deduped_ld_candidates_by_best_source_score_before_limit() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper 1184399 DN80 PN16", limit=2)

    assert [result.product["article_norm"] for result in response.results] == [
        "11100800162muld000003000",
        "11100800162muld000005000",
    ]
    assert [result.score for result in response.results] == [0.97, 0.97]


def test_search_applies_e5_query_prefixes() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="intfloat/multilingual-e5-base",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    engine.search("Temper DN80 PN16", limit=1)

    assert fake_model.calls[0]["texts"][0].startswith("query: ")


def test_search_keeps_bge_m3_queries_without_manual_prefixes() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="BAAI/bge-m3",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    engine.search("Temper DN80 PN16", limit=1)

    assert fake_model.calls[0]["texts"] == ["Temper DN80 PN16"]


def test_search_leaves_raw_query_untouched_for_default_model() -> None:
    original_query = "  ТЕМПЕР   Ду80/Ру16? Ёлка  "
    fake_model = FakeModel(calls=[])

    class RecordingQdrantClient:
        def __init__(self) -> None:
            self.query_calls: list[dict[str, object]] = []

        def query_points(self, **kwargs: object) -> object:
            self.query_calls.append(kwargs)
            return SimpleNamespace(points=[])

    fake_client = RecordingQdrantClient()
    engine = SearchEngine(
        model_name="text-embedding-3-small",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    engine.search(original_query, limit=1)

    assert fake_model.calls[0]["texts"] == [original_query]
    assert fake_client.query_calls[0]["prefetch"][1].query.text == original_query


def test_search_keeps_results_without_threshold() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper 1184399 DN80 PN16", limit=5)

    assert response.count == 3
    assert [result.rank for result in response.results] == [1, 2, 3]
    assert [result.product["article_norm"] for result in response.results] == [
        "11100800162muld000003000",
        "11100800162muld000005000",
        "11100800162muld000004000",
    ]
    assert [result.score for result in response.results] == [0.97, 0.97, 0.91]


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

    fake_model = FakeModel(calls=[])
    fake_client = ThresholdQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )
    engine.settings = SimpleNamespace(
        embedding_normalize=True,
        qdrant_dense_vector_name="dense",
        qdrant_sparse_vector_name="sparse",
        dense_score_threshold=0.75,
        bm25_score_threshold=4.0,
    )

    response = engine.search("Temper DN80 PN16", limit=1)

    query_call = fake_client.query_calls[0]
    assert query_call["prefetch"][0].score_threshold == 0.75
    assert query_call["prefetch"][1].score_threshold == 4.0
    assert response.count == 1
    assert response.results[0].score == pytest.approx(0.02)


def test_search_does_not_fallback_to_collection_from_build_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    (
        reports_dir / "index_build_baai-bge-m3.json"
    ).write_text(
        (
            '{'
            '"embedding_model":"BAAI/bge-m3",'
            '"collection_alias":"steel_products_active",'
            '"collection_name":"steel_products_baai-bge-m3_20260806T132928Z"'
            '}'
        ),
        encoding="utf-8",
    )

    fake_model = FakeModel(calls=[])
    fake_client = MissingAliasQdrantClient()
    engine = SearchEngine(
        model_name="BAAI/bge-m3",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

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
    (
        reports_dir / "index_build_baai-bge-m3.json"
    ).write_text(
        (
            '{'
            '"embedding_model":"BAAI/bge-m3",'
            '"collection_alias":"steel_products_active",'
            '"collection_name":"steel_products_baai-bge-m3_20260806T132928Z"'
            '}'
        ),
        encoding="utf-8",
    )

    fake_client = MissingAliasQdrantClient()
    engine = SearchEngine(
        model_name="BAAI/bge-m3",
        client=fake_client,
        model_factory=lambda: SimpleNamespace(get_sentence_embedding_dimension=lambda: 1024),
    )

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
def test_search_regressions_cover_expected_queries(
    query: str, expected_count: int
) -> None:
    fake_model = FakeModel(calls=[])
    fake_client = RegressionQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search(query, limit=20)

    assert response.count == expected_count
    assert len(response.results) == expected_count
    assert len({result.product["article_norm"] for result in response.results}) == expected_count
