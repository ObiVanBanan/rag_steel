from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

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
                        "connection": "фланцевое",
                        "medium": "жидкость",
                        "control": "ручное",
                        "url": "https://example.invalid/doc-1",
                        "ld_candidates": [
                            _ld_candidate(
                                "11100800162MULD000003000",
                                "11100800162muld000003000",
                                name="LD Temper DN80 PN16",
                                dn=80,
                                pn_bar=16,
                                connection="фланцевое",
                                medium="жидкость",
                                control="ручное",
                                url="https://example.invalid/ld-a",
                                price=12130,
                            ),
                            _ld_candidate(
                                "11100800162MULD000004000",
                                "11100800162muld000004000",
                                name="LD Temper DN50 PN16",
                                dn=50,
                                pn_bar=16,
                                connection="фланцевое",
                                medium="жидкость",
                                control="ручное",
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
                        "connection": "резьбовое",
                        "medium": "газ",
                        "control": "электропривод",
                        "url": "https://example.invalid/doc-2",
                        "ld_candidates": [
                            _ld_candidate(
                                "11100800162MULD000003000",
                                "11100800162muld000003000",
                                name="LD Temper DN80 PN16",
                                dn=80,
                                pn_bar=16,
                                connection="фланцевое",
                                medium="жидкость",
                                control="ручное",
                                url="https://example.invalid/ld-a",
                                price=12130,
                            ),
                            _ld_candidate(
                                "11100800162MULD000005000",
                                "11100800162muld000005000",
                                name="LD Broen DN50 PN10",
                                dn=50,
                                pn_bar=10,
                                connection="резьбовое",
                                medium="газ",
                                control="электропривод",
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

        if not normalized or "несуществующий" in normalized or "пустой запрос" in normalized:
            return SimpleNamespace(points=[])

        if any(
            token in normalized
            for token in [
                "temper",
                "broen",
                "фланцевый кран",
                "ду80 ру16",
                "dn80 pn16",
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
                        "connection": "С„Р»Р°РЅС†РµРІРѕРµ",
                        "medium": "Р¶РёРґРєРѕСЃС‚СЊ",
                        "control": "СЂСѓС‡РЅРѕРµ",
                        "url": f"https://example.invalid/source-{index}",
                        "ld_candidates": [
                            _ld_candidate(
                                candidate_article,
                                candidate_article,
                                name=f"LD {index}",
                                dn=80 if result_count == 20 else 50,
                                pn_bar=16,
                                connection="С„Р»Р°РЅС†РµРІРѕРµ",
                                medium="Р¶РёРґРєРѕСЃС‚СЊ",
                                control="СЂСѓС‡РЅРѕРµ",
                                url=f"https://example.invalid/ld-{index}",
                                price=1000 + index,
                            )
                        ],
                    },
                )
            )
        return SimpleNamespace(points=points)


def test_search_deduplicates_ld_candidates_and_builds_evidence() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper 1184399 Ду80 Ру16", limit=5)

    assert isinstance(response, SearchResponse)
    assert response.query == "Temper 1184399 Ду80 Ру16"
    assert response.count == 3
    assert response.processed_query.possible_article_tokens == ["1184399"]
    assert [result.rank for result in response.results] == [1, 2, 3]
    assert len({result.product["article_norm"] for result in response.results}) == 3

    top = response.results[0]
    assert top.id == "11100800162muld000003000"
    assert top.product["article"] == "11100800162MULD000003000"
    assert top.product["article_norm"] == "11100800162muld000003000"
    assert top.relevance_rating == pytest.approx(round(top.score * 100, 1))
    assert top.score_breakdown["ld_field_score"] == 1.0
    assert top.score_breakdown["final_score"] == pytest.approx(top.score)
    assert top.score_breakdown["final_score"] == pytest.approx(
        (top.score_breakdown["source_score"] * 0.70) + 0.30
    )
    assert len(top.source_evidence) == 2
    assert [item["source_article"] for item in top.source_evidence] == [
        "1184399",
        "a0486",
    ]
    assert [item["source_rank"] for item in top.source_evidence] == [1, 2]
    assert all(item["source_score"] is not None for item in top.source_evidence)
    assert "Совпадает DN 80" in top.match_reasons
    assert "Совпадает PN 16" in top.match_reasons
    assert top.mismatches == []

    second = response.results[1]
    assert second.product["article_norm"] == "11100800162muld000004000"
    assert second.score_breakdown["final_score"] <= top.score_breakdown["final_score"]

    third = response.results[2]
    assert third.product["article_norm"] == "11100800162muld000005000"

    assert "normalize" in response.timing_ms
    assert "embedding" in response.timing_ms
    assert "qdrant" in response.timing_ms
    assert "ranking" in response.timing_ms
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["texts"] == [response.processed_query.semantic_text]
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
    assert query_call["query"].fusion == "rrf"


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


@pytest.mark.parametrize(
    ("query", "expected_count"),
    [
        ("1184399", 7),
        ("а0486", 7),
        ("А0486", 7),
        ("а-0486", 7),
        ("КШ.П.П.015.40-01", 7),
        ("кшпп0154001", 7),
        ("Temper DN80 PN16", 20),
        ("Temper 1184399 Ду80 Ру16", 20),
        ("Broen Ду80 Ру16", 20),
        ("фланцевый кран Ду50 Ру40", 20),
        ("несуществующий артикул", 0),
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
