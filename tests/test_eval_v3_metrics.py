from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval import evaluate_deepseek_v3, evaluate_e2e_v3, evaluate_rag_v3
from eval.v3_common import compare_expected_actual, hard_exact_match
from eval.v3_schema import ExpectedAttributes


def _deepseek_case(
    *,
    expected: dict[str, object] | None = None,
    actual: dict[str, object] | None = None,
    brand_expected: str | None = "Temper",
    brand_actual: str | None = "Temper",
    invalid_response: bool = False,
) -> evaluate_deepseek_v3.DeepSeekCaseResult:
    expected = expected or {"brand": "Temper", "dn": 80, "pn_bar": 16, "connection": "сварное"}
    actual = actual or expected
    return evaluate_deepseek_v3.DeepSeekCaseResult(
        id="case",
        query="q",
        category="c",
        expected_status="exact_match",
        expected=expected,
        actual=actual,
        brand_expected=brand_expected,
        brand_actual=brand_actual,
        brand_correct=brand_expected == brand_actual,
        comparison=compare_expected_actual(
            ExpectedAttributes.model_validate(expected),
            ExpectedAttributes.model_validate(actual),
        ),
        invalid_response=invalid_response,
        latency_ms=10.0,
    )


def test_deepseek_field_and_missing_rates() -> None:
    cases = [
        _deepseek_case(),
        _deepseek_case(expected={"brand": "Temper", "dn": 80, "pn_bar": 16, "connection": "сварное"}, actual={"brand": "Temper", "dn": 50, "pn_bar": 16, "connection": "сварное"}),
        _deepseek_case(expected={"brand": "Temper", "dn": 80, "pn_bar": 16, "connection": "сварное"}, actual={"brand": "Temper", "dn": 80, "pn_bar": None, "connection": "сварное"}),
        _deepseek_case(expected={"brand": "Temper", "dn": None, "pn_bar": 16, "connection": None}, actual={"brand": "Temper", "dn": 80, "pn_bar": 16, "connection": None}),
    ]

    assert evaluate_deepseek_v3._field_accuracy(cases, "dn") == pytest.approx(2 / 3)
    assert evaluate_deepseek_v3._hallucination_rate(cases, ("brand", "dn", "pn_bar", "connection")) == pytest.approx(1 / 4)
    assert evaluate_deepseek_v3._missing_rate(cases, ("brand", "dn", "pn_bar", "connection")) == pytest.approx(1 / 4)
    assert hard_exact_match(
        ExpectedAttributes(brand="Temper", dn=80, pn_bar=16, connection="сварное"),
        ExpectedAttributes(brand="Temper", dn=80, pn_bar=16, connection="сварное"),
    )


def test_rag_ranking_metrics_cover_hit_precision_and_mrr() -> None:
    returned = ["a", "b", "c", "d", "e"]
    target = ["c", "x"]

    assert evaluate_rag_v3._percent_hit(returned, target, 5) is True
    assert evaluate_rag_v3._percent_hit(returned, ["x"], 5) is False
    assert evaluate_rag_v3._precision_at_k(returned, target, 5) == pytest.approx(1 / 5)
    assert evaluate_rag_v3._mrr(returned, target) == pytest.approx(1 / 3)
    assert evaluate_rag_v3._coverage_at_k(returned, target, 5) == pytest.approx(1 / 2)


def test_e2e_failure_stage_priority() -> None:
    expected = ExpectedAttributes(brand="Temper", dn=80, pn_bar=16, connection="сварное")
    response = SimpleNamespace(status="exact_match")

    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="not_found",
                expected_attributes=expected,
                eligible_competitor_articles=[],
                preferred_competitor_articles=[],
            ),
            response=SimpleNamespace(status="exact_match"),
            actual_requested=expected,
            hard_violation=False,
            eligible_hit_at_5=False,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "status_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=ExpectedAttributes(brand="Broen", dn=80, pn_bar=16, connection="сварное"),
            hard_violation=False,
            eligible_hit_at_5=False,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "brand_gate_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=ExpectedAttributes(brand="Temper", dn=50, pn_bar=16, connection="сварное"),
            hard_violation=False,
            eligible_hit_at_5=False,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "deepseek_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=expected,
            hard_violation=True,
            eligible_hit_at_5=False,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "hard_filter_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=expected,
            hard_violation=False,
            eligible_hit_at_5=False,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "retrieval_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=expected,
            hard_violation=False,
            eligible_hit_at_5=True,
            preferred_hit_at_5=False,
            ld_mapping_ok=True,
        )
        == "ranking_failure"
    )
    assert (
        evaluate_e2e_v3._classify_failure_stage(
            SimpleNamespace(
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=["a"],
                preferred_competitor_articles=["a"],
            ),
            response=response,
            actual_requested=expected,
            hard_violation=False,
            eligible_hit_at_5=True,
            preferred_hit_at_5=True,
            ld_mapping_ok=False,
        )
        == "ld_mapping_failure"
    )
