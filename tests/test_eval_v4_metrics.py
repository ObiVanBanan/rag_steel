from __future__ import annotations

import json
from types import SimpleNamespace

from eval.compare_v4_results import compare_v4_results
from eval.evaluate_deepseek_v4 import _compare_expected_actual
from eval.evaluate_rag_v4 import _DummyEmbedder, _hard_violation
from eval.evaluate_resolution_v4 import (
    ResolutionCaseResult,
    _evaluate_case,
    _false_correction_rate,
)
from eval.v4_schema import EvalCase, ExpectedAttributes


def test_rag_hard_violation_treats_pn_as_minimum_pressure() -> None:
    expected = ExpectedAttributes(
        resolved_brand="Temper",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )
    higher_pn_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    brand="Temper",
                    dn=50,
                    pn_bar=25,
                    connection="flanged",
                )
            )
        ]
    )
    lower_pn_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    brand="Temper",
                    dn=50,
                    pn_bar=10,
                    connection="flanged",
                )
            )
        ]
    )

    assert _hard_violation(expected, higher_pn_response) is False
    assert _hard_violation(expected, lower_pn_response) is True


def test_dummy_embedder_uses_its_full_dimension() -> None:
    embedder = _DummyEmbedder(dimension=7)

    assert len(embedder.embed_query("test")) == 7
    assert len(embedder.embed_documents(["a", "b"])[0]) == 7


def test_resolution_false_correction_ignores_identity_conflict_with_valid_brand() -> None:
    case = ResolutionCaseResult(
        id="case-1",
        query="Temper A-001",
        category="brand_article_conflict",
        expected_status="not_found",
        expected_resolution_mode="identity_conflict",
        expected={
            "raw_brand": "Temper",
            "resolved_brand": "Temper",
            "article": "A-001",
            "resolved_article": None,
        },
        actual={
            "raw_brand": "Temper",
            "resolved_brand": "Temper",
            "article": "A-001",
            "resolved_article": None,
        },
        actual_status="not_found",
        actual_resolution_mode="identity_conflict",
        reason_code="IDENTITY_CONFLICT",
        comparison={
            "wrong_fields": [],
            "hallucinated_fields": [],
            "missing_fields": [],
        },
        latency_ms=1.0,
    )

    assert _false_correction_rate([case]) == 0.0


def test_resolution_false_correction_counts_hallucinated_identity() -> None:
    case = ResolutionCaseResult(
        id="case-2",
        query="unknown thing",
        category="unknown_article",
        expected_status="not_found",
        expected_resolution_mode="article_not_found",
        expected={
            "raw_brand": None,
            "resolved_brand": None,
            "article": "ZZ-UNKNOWN",
            "resolved_article": None,
        },
        actual={
            "raw_brand": None,
            "resolved_brand": "Temper",
            "article": "ZZ-UNKNOWN",
            "resolved_article": "A-001",
        },
        actual_status="exact_match",
        actual_resolution_mode="article_exact",
        reason_code=None,
        comparison={
            "wrong_fields": [],
            "hallucinated_fields": ["resolved_brand", "resolved_article"],
            "missing_fields": [],
        },
        latency_ms=1.0,
    )

    assert _false_correction_rate([case]) == 1.0


def test_resolution_v4_brand_only_hard_mismatch_counts_as_not_found() -> None:
    case = EvalCase(
        id="case-3",
        category="v3_regression",
        query="Temper DN999 PN999",
        expected_status="not_found",
        expected_resolution_mode="brand_exact",
        expected_attributes=ExpectedAttributes(
            raw_brand="Temper",
            resolved_brand="Temper",
            article=None,
            resolved_article=None,
            dn=999,
            pn_bar=999,
        ),
        eligible_competitor_articles=[],
        preferred_competitor_articles=[],
        expected_ld_articles_by_competitor={},
    )

    engine = SimpleNamespace(
        query_resolver=SimpleNamespace(
            resolve=lambda **_: SimpleNamespace(
                brand=SimpleNamespace(raw="Temper", canonical="Temper"),
                article=SimpleNamespace(raw=None, article=None),
                resolution_mode="brand_exact",
                reason_code=None,
            )
        )
    )

    result = _evaluate_case(engine, case)

    assert result.actual_status == "not_found"
    assert result.actual_resolution_mode == "brand_exact"


def test_deepseek_comparison_ignores_resolution_fields() -> None:
    expected = ExpectedAttributes(
        raw_brand="Temper",
        article="A-001",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )
    actual = ExpectedAttributes(
        raw_brand="Temper",
        resolved_brand="Broen",
        article="A-001",
        resolved_article="A-001",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )

    comparison = _compare_expected_actual(expected, actual)

    assert comparison == {
        "wrong_fields": [],
        "hallucinated_fields": [],
        "missing_fields": [],
    }


def test_compare_v4_results_rejects_mismatched_case_counts(tmp_path) -> None:
    deepseek = tmp_path / "deepseek.json"
    resolution = tmp_path / "resolution.json"
    rag = tmp_path / "rag.json"
    e2e = tmp_path / "e2e.json"
    output = tmp_path / "summary.md"

    payload_160 = {
        "summary": {"cases": 160},
        "dataset_sha256": "same",
    }
    payload_1 = {
        "summary": {"cases": 1},
        "dataset_sha256": "same",
    }
    for path, payload in (
        (deepseek, payload_160),
        (resolution, payload_160),
        (rag, payload_1),
        (e2e, payload_160),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        compare_v4_results(
            deepseek_path=deepseek,
            resolution_path=resolution,
            rag_path=rag,
            e2e_path=e2e,
            output_path=output,
        )
    except ValueError as exc:
        assert "mismatched case counts" in str(exc)
    else:
        raise AssertionError("compare_v4_results should reject mismatched case counts")
