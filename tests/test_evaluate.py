from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import evaluate


class _FakeEmbeddingModel:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def test_metric_helpers_cover_rank_and_distribution() -> None:
    assert evaluate._ndcg_at_k(["a", "b"], {"a"}, 20) == pytest.approx(1.0)
    assert evaluate._ndcg_at_k(["b", "a"], {"a"}, 20) == pytest.approx(1 / 1.5849625007)
    assert evaluate._reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)
    assert evaluate._percentile([5.0], 95) == pytest.approx(5.0)
    assert evaluate._percentile([1.0, 3.0, 5.0], 50) == pytest.approx(3.0)


def test_load_queries_parses_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "queries.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "Temper DN80 PN16",
                        "category": "brand_dn_pn",
                        "expected_ld_articles": ["a", "b"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "query": "No match",
                        "category": "no_match",
                        "expected_ld_articles": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = evaluate._load_queries(dataset_path)

    assert [record.query for record in records] == ["Temper DN80 PN16", "No match"]
    assert records[0].expected_ld_articles == ["a", "b"]
    assert records[1].category == "no_match"


class _FakeClient:
    def __init__(self) -> None:
        self.count_calls: list[dict[str, object]] = []

    def count(self, **kwargs: object) -> object:
        self.count_calls.append(kwargs)
        return SimpleNamespace(count=42)


class _FakeEngine:
    def __init__(self, *, model_name: str, **_: object) -> None:
        self.model_name = model_name

    def search(self, query: str, limit: int = 20, **_: object) -> SimpleNamespace:
        expected_article = "gold-a" if query == "Temper DN80 PN16" else "gold-b"
        if self.model_name == "paraphrase-multilingual-MiniLM-L12-v2":
            articles = [expected_article, "decoy"]
            timing = 10.0
        else:
            articles = ["decoy", expected_article]
            timing = 30.0
        if query == "No match":
            articles = ["decoy", "decoy-2"]
        return SimpleNamespace(
            results=[
                SimpleNamespace(product={"article_norm": article})
                for article in articles[:limit]
            ],
            timing_ms={"total": timing},
            query=query,
        )


def test_compare_models_ranks_by_relevance_and_ignores_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "queries.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "Temper DN80 PN16",
                        "category": "brand_dn_pn",
                        "expected_ld_articles": ["gold-a"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "query": "Broen DN50 PN10",
                        "category": "brand_dn_pn",
                        "expected_ld_articles": ["gold-b"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "query": "No match",
                        "category": "no_match",
                        "expected_ld_articles": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_client_factory(_: str) -> _FakeClient:
        return _FakeClient()

    fake_registry = {
        "paraphrase-multilingual-MiniLM-L12-v2": lambda: _FakeEmbeddingModel(384),
        "intfloat/multilingual-e5-base": lambda: _FakeEmbeddingModel(768),
    }

    def fake_build_index_fn(
        _: Path,
        *,
        model_name: str,
        recreate: bool,
        client: object,
        metadata_path: Path,
        model_factory: object,
    ) -> SimpleNamespace:
        assert recreate is False
        assert callable(model_factory)
        assert isinstance(client, _FakeClient)
        assert metadata_path.name.startswith("index_build_")
        return SimpleNamespace(
            metadata=SimpleNamespace(
                collection_name=f"collection-{model_name.replace('/', '_')}",
                document_count=2,
                embedding_dimension=fake_registry[model_name]().get_sentence_embedding_dimension(),
            )
        )

    monkeypatch.setattr(evaluate, "SearchEngine", _FakeEngine)
    monkeypatch.setattr(evaluate, "MODEL_REGISTRY", fake_registry)

    results = evaluate.compare_models(
        models=[
            "paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-base",
        ],
        dataset_path=dataset_path,
        source_csv=tmp_path / "mapping_results.csv",
        qdrant_url="http://example.invalid",
        limit=2,
        client_factory=fake_client_factory,
        build_index_fn=fake_build_index_fn,
    )

    assert [result.model_name for result in results] == [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-base",
    ]
    assert [result.status for result in results] == ["completed", "completed"]
    assert results[0].top_k == 2
    assert results[0].recall_at_k == pytest.approx(1.0)
    assert results[0].ndcg_at_k == pytest.approx(1.0)
    assert results[0].precision_at_k == pytest.approx(0.5)
    assert results[0].mrr == pytest.approx(1.0)
    assert results[0].cold_query_latency_ms == pytest.approx(10.0)
    assert results[0].latency_p50_ms == pytest.approx(10.0)
    assert results[0].no_match_false_positive_rate == pytest.approx(1.0)
    assert results[1].ndcg_at_k < results[0].ndcg_at_k
    assert results[1].recall_at_k == pytest.approx(1.0)
    assert results[0].index_size_points == 42
    assert results[0].per_category_recall_at_k["brand_dn_pn"] == pytest.approx(1.0)
    assert results[0].query_examples[0]["query"] == "Temper DN80 PN16"
    assert results[0].query_examples[0]["returned_ld_articles"] == ["gold-a", "decoy"]

    report_path = tmp_path / "model_comparison.md"
    report = evaluate.render_report(
        results,
        dataset_path=dataset_path,
        output_path=report_path,
    )

    assert report_path.read_text(encoding="utf-8") == report
    assert "Selected Model" in report
    assert "`paraphrase-multilingual-MiniLM-L12-v2`" in report
    assert "| Model | nDCG@2 | Recall@2 | MRR | Precision@2 |" in report

    results_json_path = tmp_path / "results.json"
    examples_json_path = tmp_path / "results_examples.json"
    examples_payload = evaluate.write_examples_json(
        results,
        output_path=examples_json_path,
        run_id="20260804T000000Z",
    )
    payload = evaluate.write_results_json(
        results,
        dataset_path=dataset_path,
        source_csv=tmp_path / "mapping_results.csv",
        qdrant_url="http://example.invalid",
        models=[
            "paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-base",
        ],
        output_path=results_json_path,
        run_id="20260804T000000Z",
        examples_path=examples_json_path,
    )
    assert results_json_path.exists()
    assert examples_json_path.exists()
    assert payload["selected_model"] == "paraphrase-multilingual-MiniLM-L12-v2"
    assert payload["examples_path"] == str(examples_json_path)
    assert payload["results"][0]["status"] == "completed"
    assert examples_payload["models"][0]["query_examples"][0]["query"] == "Temper DN80 PN16"


def test_render_report_handles_all_failed_models(tmp_path: Path) -> None:
    results = [
        evaluate._failed_result(
            "paraphrase-multilingual-MiniLM-L12-v2",
            RuntimeError("boom"),
        )
    ]
    report_path = tmp_path / "model_comparison.md"

    report = evaluate.render_report(
        results,
        dataset_path=tmp_path / "queries.jsonl",
        output_path=report_path,
    )

    assert "No models completed successfully." in report
    assert "## Failed Models" in report
    assert "`paraphrase-multilingual-MiniLM-L12-v2`: RuntimeError: boom" in report
