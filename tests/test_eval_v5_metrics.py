from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval.build_v5_eval_dataset import _to_v5_expected
from eval.compare_v5_results import compare_v5_results
from eval.evaluate_deepseek_v5 import evaluate_deepseek_v5
from eval.v4_schema import ExpectedAttributes as V4ExpectedAttributes
from eval.v5_schema import EvalCase, ExpectedAttributes


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
