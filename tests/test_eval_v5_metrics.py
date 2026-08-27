from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval.build_v5_eval_dataset import _to_v5_expected, _transform_v4_records
from eval.compare_v5_results import compare_v5_results
from eval.evaluate_deepseek_v5 import evaluate_deepseek_v5
from eval.evaluate_resolution_v5 import (
    ResolutionCaseResult,
    _build_family_summary,
    render_report,
)
from eval.v4_schema import EvalCase as V4EvalCase
from eval.v4_schema import ExpectedAttributes as V4ExpectedAttributes
from eval.v5_schema import EvalCase, ExpectedAttributes
from rag_steel.schemas import LDProduct, SteelProductDocument


class _FakeExtractor:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def extract(self, query: str) -> SimpleNamespace:
        del query
        return SimpleNamespace(**self.payload)


def test_evaluate_deepseek_v5_scores_semantic_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "v5.jsonl"
    cases = [
        EvalCase(
            id="case-1",
            category="brand_semantic",
            query="Темпер DN50",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=ExpectedAttributes(
                brand="Temper",
                resolved_brand="Temper",
                dn=50,
            ),
        ),
        EvalCase(
            id="case-2",
            category="unsupported_brand",
            query="NoSuchBrand DN50",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=ExpectedAttributes(
                brand=None,
                resolved_brand=None,
                dn=50,
            ),
        ),
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )

    payload = evaluate_deepseek_v5(
        dataset_path=dataset_path,
        extractor_factory=lambda _settings: _FakeExtractor(
            {
                "brand": "Temper",
                "article": None,
                "dn": 50,
                "pn_bar": None,
                "connection": None,
                "body_material": None,
                "medium": None,
                "control": None,
                "temperature": None,
                "length_mm": None,
                "series": None,
            }
        ),
        settings=SimpleNamespace(deepseek_model="fake"),
    )

    assert payload["summary"]["cases"] == 2
    assert payload["summary"]["brand_interpretation_accuracy"] == 1.0
    assert payload["summary"]["dn_interpretation_accuracy"] == 1.0
    assert payload["summary"]["technical_error_rate"] == 0.0


def test_compare_v5_results_uses_new_metric_names(tmp_path: Path) -> None:
    payload = {
        "summary": {
            "cases": 2,
            "brand_interpretation_accuracy": 1.0,
            "article_identity_accuracy": 1.0,
            "hard_interpretation_exact_match": 1.0,
            "overall_resolution_accuracy": 1.0,
            "false_correction_rate": 0.0,
            "status_accuracy": 1.0,
            "hard_violation_rate": 0.0,
            "requested_contract_accuracy": 1.0,
            "ld_mapping_exact_rate": 1.0,
            "overall_pass_rate": 1.0,
        },
        "dataset_sha256": "abc",
    }
    for name in ("deepseek", "resolution", "rag", "e2e"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    report = compare_v5_results(
        deepseek_path=tmp_path / "deepseek.json",
        resolution_path=tmp_path / "resolution.json",
        rag_path=tmp_path / "rag.json",
        e2e_path=tmp_path / "e2e.json",
        output_path=tmp_path / "summary.md",
    )

    assert "DeepSeek brand interpretation accuracy" in report
    assert "E2E requested contract accuracy" in report


def test_v5_expected_preserves_conflict_resolved_article() -> None:
    expected = V4ExpectedAttributes(
        raw_brand="Temper",
        resolved_brand="Temper",
        article="A-123",
        resolved_article=None,
    )

    transformed = _to_v5_expected(expected)

    assert transformed.article == "A-123"
    assert transformed.resolved_article is None


def test_transform_v4_records_derives_family_from_preferred_articles() -> None:
    record = V4EvalCase(
        id="v4_case_1",
        category="mixed_semantic",
        query="нужен затвор PALUR",
        expected_status="exact_match",
        expected_resolution_mode="brand_exact",
        expected_attributes=V4ExpectedAttributes(
            raw_brand="PALUR",
            resolved_brand="PALUR",
            article=None,
            resolved_article=None,
        ),
        preferred_competitor_articles=["PALUR-200"],
        eligible_competitor_articles=[],
        expected_ld_articles_by_competitor={},
    )
    document = SteelProductDocument(
        steel_id="butter-1",
        article="PALUR-200",
        article_norm="palur-200",
        article_compact="palur200",
        name="Затвор дисковый поворотный PALUR-200",
        name_variants=["Затвор дисковый поворотный PALUR-200"],
        brand="PALUR",
        dn=200,
        pn_bar=16,
        connection="фланцевое",
        body_material="сталь 20",
        medium=None,
        control=None,
        temperature=None,
        length_mm=None,
        url=None,
        semantic_text="",
        lexical_text="",
        ld_candidates=[
            LDProduct(
                article="LD-BUTTER",
                article_norm="ld-butter",
                name="LD Butterfly",
                url=None,
                dn=200,
                pn_bar=16,
                connection="фланцевое",
            )
        ],
    )

    transformed = _transform_v4_records([record], [document])

    assert transformed[0].product_family == "butterfly_valve"
    assert transformed[0].search_intent == "description"


def test_evaluate_deepseek_v5_rejects_dropped_hard_constraints(tmp_path: Path) -> None:
    dataset_path = tmp_path / "v5.jsonl"
    case = EvalCase(
        id="case-1",
        category="v3_regression",
        query="Temper DN107 PN999",
        expected_status="not_found",
        expected_resolution_mode="brand_exact",
        expected_attributes=ExpectedAttributes(
            brand="Temper",
            resolved_brand="Temper",
            dn=100,
            pn_bar=999,
        ),
    )
    dataset_path.write_text(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    payload = evaluate_deepseek_v5(
        dataset_path=dataset_path,
        extractor_factory=lambda _settings: _FakeExtractor(
            {
                "brand": "Temper",
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
        ),
        settings=SimpleNamespace(deepseek_model="fake"),
    )

    assert payload["summary"]["cases"] == 1
    assert payload["summary"]["brand_interpretation_accuracy"] == 1.0
    assert payload["summary"]["dn_interpretation_accuracy"] == 0.0
    assert payload["summary"]["pn_interpretation_accuracy"] == 0.0
    assert payload["summary"]["hard_interpretation_exact_match"] == 0.0
    assert payload["failures"][0]["missing_fields"] == ["dn", "pn_bar"]


def test_resolution_v5_family_summary_separates_article_and_description() -> None:
    cases = [
        ResolutionCaseResult(
            id="a1",
            query="ART-1",
            category="article_only_exact",
            expected_status="exact_match",
            expected_resolution_mode="article_exact",
            expected={},
            actual={},
            actual_status="exact_match",
            actual_resolution_mode="article_exact",
            reason_code=None,
            product_family="ball_valve",
            search_intent="article",
            comparison={"wrong_fields": [], "missing_fields": [], "hallucinated_fields": []},
            latency_ms=1.0,
        ),
        ResolutionCaseResult(
            id="d1",
            query="DN50 PN16",
            category="mixed_semantic",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected={},
            actual={},
            actual_status="not_found",
            actual_resolution_mode="brand_exact",
            reason_code="ARTICLE_NOT_FOUND",
            product_family="ball_valve",
            search_intent="description",
            comparison={"wrong_fields": ["brand"], "missing_fields": [], "hallucinated_fields": []},
            latency_ms=1.0,
        ),
    ]

    summary = _build_family_summary(cases)

    assert summary["ball_valve"]["cases"] == 2
    assert summary["ball_valve"]["article_cases"] == 1
    assert summary["ball_valve"]["description_cases"] == 1
    assert summary["ball_valve"]["article_accuracy"] == 1.0
    assert summary["ball_valve"]["description_accuracy"] == 0.0
    assert summary["ball_valve"]["unexpected_not_found_rate"] == 0.5


def test_resolution_v5_report_includes_product_family_table() -> None:
    payload = {
        "summary": {
            "cases": 2,
            "brand_exact_accuracy": 1.0,
            "brand_fuzzy_accuracy": 0.0,
            "article_exact_accuracy": 1.0,
            "article_fuzzy_accuracy": 0.0,
            "article_resolution_accuracy": 1.0,
            "description_resolution_accuracy": 0.5,
            "ambiguity_accuracy": 1.0,
            "ambiguous_rate": 0.0,
            "identity_conflict_accuracy": 1.0,
            "overall_resolution_accuracy": 0.75,
            "unexpected_not_found_rate": 0.25,
            "false_correction_rate": 0.0,
        },
        "by_resolution_mode": {"article_exact": 1, "brand_exact": 1},
        "by_product_family": {
            "ball_valve": {
                "cases": 2,
                "article_cases": 1,
                "description_cases": 1,
                "article_accuracy": 1.0,
                "description_accuracy": 0.5,
                "overall_accuracy": 0.75,
                "unexpected_not_found_rate": 0.25,
            }
        },
    }

    report = render_report(payload, Path("C:/Users/theso/Desktop/job/rag_steel/.report_test.md"))

    assert "## By Product Family" in report
    assert "| ball_valve | 2 | 1 | 1 | 1.0000 | 0.5000 | 0.7500 | 0.2500 |" in report
