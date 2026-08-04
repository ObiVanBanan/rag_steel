from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

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
                        "name": "Кран шаровой Temper Ду80 Ру16",
                        "brand": "Temper",
                        "dn": 80,
                        "pn_bar": 16,
                        "connection": "фланцевое",
                        "medium": "жидкость",
                        "control": "ручное",
                        "url": "https://example.invalid/doc-1",
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
                        "name": "Кран шаровой Broen Ду50 Ру10",
                        "brand": "Broen",
                        "dn": 50,
                        "pn_bar": 10,
                        "connection": "резьбовое",
                        "medium": "газ",
                        "control": "электропривод",
                        "url": "https://example.invalid/doc-2",
                    },
                ),
            ]
        )


def test_search_reorders_candidates_by_structured_rerank() -> None:
    fake_model = FakeModel(calls=[])
    fake_client = FakeQdrantClient()
    engine = SearchEngine(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        client=fake_client,
        model_factory=lambda: fake_model,
    )

    response = engine.search("Temper 1184399 Ду80 Ру16", limit=2)

    assert isinstance(response, SearchResponse)
    assert response.query == "Temper 1184399 Ду80 Ру16"
    assert response.count == 2
    assert response.processed_query.possible_article_tokens == ["1184399"]
    assert response.results[0].rank == 1
    assert response.results[0].id == "doc-1"
    assert response.results[0].score == 0.91
    assert response.results[0].hybrid_score == pytest.approx(0.91 / 0.97)
    assert response.results[0].payload["article"] == "1184399"
    assert response.results[0].product["article"] == "1184399"
    assert response.results[0].product["dn"] == 80
    assert response.results[0].source_evidence == [
        {"article": "1184399", "name": "Кран шаровой Temper Ду80 Ру16"}
    ]
    assert "Совпадает DN 80" in response.results[0].match_reasons
    assert "Совпадает PN 16" in response.results[0].match_reasons
    assert "Совпадает бренд Temper" in response.results[0].match_reasons
    assert response.results[0].mismatches == []
    assert response.results[0].score_breakdown["hybrid_score"] == pytest.approx(0.91 / 0.97)
    assert response.results[0].score_breakdown["text_exactness"] == 1.0
    assert response.results[0].score_breakdown["source_field_score"] == 1.0
    first_score = response.results[0].score_breakdown["source_score"]
    second_score = response.results[1].score_breakdown["source_score"]
    assert first_score > second_score
    assert "normalize" in response.timing_ms
    assert "embedding" in response.timing_ms
    assert "qdrant" in response.timing_ms
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
