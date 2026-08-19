from __future__ import annotations

import json
from types import SimpleNamespace

from eval.compare_v4_results import compare_v4_results
from eval.evaluate_deepseek_v4 import _compare_expected_actual
from eval.evaluate_e2e_v4 import _evaluate_case as _evaluate_e2e_case
from eval.evaluate_rag_v4 import (
    RagCaseResult,
    _article_comparison_key,
    _build_category_summary,
    _compare_requested,
    _DummyEmbedder,
    _eligible_hit_5_failures,
    _hard_violation,
    _ld_mapping_exact,
    _percent_hit,
)
from eval.evaluate_rag_v4 import _evaluate_case as _evaluate_rag_case
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


def test_rag_article_comparison_key_normalizes_norm_and_compact_forms() -> None:
    assert _article_comparison_key("107-5450") == _article_comparison_key("1075450")


def test_rag_article_comparison_key_normalizes_punctuation_and_case() -> None:
    assert _article_comparison_key("2ЦП.00.0.016.020") == _article_comparison_key(
        "2цп000016020"
    )


def test_rag_article_comparison_key_distinguishes_different_articles() -> None:
    assert _article_comparison_key("107-5450") != _article_comparison_key("107-5451")


def test_rag_percent_hit_uses_canonical_article_identity() -> None:
    returned = ["1075450", "other"]
    target = ["107-5450"]

    assert _percent_hit(returned, target, 1) is True
    assert _percent_hit(["1075451"], target, 1) is False


def test_rag_ld_mapping_exact_normalizes_keys() -> None:
    expected = {"КШ.Ф.П.200.25-01": ["LD-1", "LD-2"]}
    returned = {"кш.ф.п.200.25-01": ["LD-2", "LD-1"]}

    assert _ld_mapping_exact(expected, returned) is True


def test_rag_ld_mapping_exact_detects_wrong_ld_list() -> None:
    expected = {"КШ.Ф.П.200.25-01": ["LD-1", "LD-2"]}
    returned = {"кш.ф.п.200.25-01": ["LD-1", "LD-3"]}

    assert _ld_mapping_exact(expected, returned) is False


def test_rag_ld_mapping_exact_keeps_distinct_article_variants_separate() -> None:
    expected = {"КШ.Ф.П.200.25-01": ["LD-MGLD"]}
    returned = {"КШ.ФП.200.25-01": ["LD-MGLD"]}

    assert _ld_mapping_exact(expected, returned) is False


def test_rag_case_result_reports_best_rank_diagnostics() -> None:
    case = EvalCase(
        id="case-4",
        category="article_only_exact",
        query="107-5450",
        expected_status="exact_match",
        expected_resolution_mode="article_exact",
        expected_attributes=ExpectedAttributes(
            resolved_brand="FORTECA",
            article="1075450",
            resolved_article="1075450",
        ),
        eligible_competitor_articles=["1075450"],
        preferred_competitor_articles=["1075450"],
        expected_ld_articles_by_competitor={"1075450": ["LD-1", "LD-2"]},
    )
    response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"raw_brand": None, "article": "107-5450"},
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
        results=[
            SimpleNamespace(
                    rank=1,
                    competitor=SimpleNamespace(
                        article="1075450",
                        brand="FORTECA",
                        dn=None,
                        pn_bar=None,
                        connection=None,
                    ),
                ld_articles=["LD-2", "LD-1"],
                product={"article": "1075450"},
                score=0.99,
            )
        ],
    )
    engine = SimpleNamespace(search_v2=lambda query, limit: response, attribute_extractor=None)

    result = _evaluate_rag_case(engine, case)

    assert result.eligible_hit_at_5 is True
    assert result.preferred_hit_at_5 is True
    assert result.eligible_best_rank == 1
    assert result.preferred_best_rank == 1
    assert result.ld_mapping_exact_rate is True


def test_rag_case_result_reports_missing_best_rank_when_absent() -> None:
    case = EvalCase(
        id="case-5",
        category="article_only_exact",
        query="107-5450",
        expected_status="exact_match",
        expected_resolution_mode="article_exact",
        expected_attributes=ExpectedAttributes(
            resolved_brand="FORTECA",
            article="107-5450",
            resolved_article="107-5450",
        ),
        eligible_competitor_articles=["107-5450"],
        preferred_competitor_articles=["107-5450"],
        expected_ld_articles_by_competitor={},
    )
    response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"raw_brand": None, "article": "107-5450"},
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
        results=[
            SimpleNamespace(
                rank=1,
                competitor=SimpleNamespace(
                    article="1075451",
                    brand="FORTECA",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=[],
                product={"article": "1075451"},
                score=0.5,
            )
        ],
    )
    engine = SimpleNamespace(search_v2=lambda query, limit: response, attribute_extractor=None)

    result = _evaluate_rag_case(engine, case)

    assert result.eligible_hit_at_5 is False
    assert result.eligible_best_rank is None
    assert result.preferred_best_rank is None


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


def test_requested_contract_compares_raw_article_and_canonical_resolution() -> None:
    expected = ExpectedAttributes(
        raw_brand="Temper",
        resolved_brand="Temper",
        article="107 5450",
        resolved_article="107-5450",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )
    actual = ExpectedAttributes(
        raw_brand="temper",
        resolved_brand="Temper",
        article="107 5450",
        resolved_article="1075450",
        dn=50,
        pn_bar=16,
        connection="Фланцевое",
    )

    comparison = _compare_requested(expected, actual)

    assert comparison == {
        "wrong_fields": [],
        "hallucinated_fields": [],
        "missing_fields": [],
    }


def test_requested_contract_detects_raw_article_regression() -> None:
    expected = ExpectedAttributes(
        raw_brand="Temper",
        resolved_brand="Temper",
        article="107 5450",
        resolved_article="107-5450",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )
    actual = ExpectedAttributes(
        raw_brand="Temper",
        resolved_brand="Temper",
        article="107-5450",
        resolved_article="1075450",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )

    comparison = _compare_requested(expected, actual)

    assert comparison["wrong_fields"] == ["article"]


def test_e2e_overall_pass_rejects_ld_and_retrieval_misses() -> None:
    case = EvalCase(
        id="case-6",
        category="article_only_normalized",
        query="107 5450",
        expected_status="exact_match",
        expected_resolution_mode="article_exact",
        expected_attributes=ExpectedAttributes(
            raw_brand="FORTECA",
            resolved_brand="FORTECA",
            article="107 5450",
            resolved_article="107-5450",
        ),
        eligible_competitor_articles=["107-5450"],
        preferred_competitor_articles=["107-5450"],
        expected_ld_articles_by_competitor={"107-5450": ["LD-1"]},
    )
    good_response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"brand": "FORTECA", "article": "107 5450", "resolved_article": "107-5450"},
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    article="107-5450",
                    brand="FORTECA",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=["LD-1"],
            )
        ],
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
    )
    hard_violation_response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"brand": "FORTECA", "article": "107 5450", "resolved_article": "107-5450"},
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    article="107-5450",
                    brand="BROEN",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=["LD-1"],
            )
        ],
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
    )
    eligible_miss_response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"brand": "FORTECA", "article": "107 5450", "resolved_article": "107-5450"},
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    article="other",
                    brand="FORTECA",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=["LD-2"],
            )
        ],
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
    )
    ld_mismatch_response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"brand": "FORTECA", "article": "107 5450", "resolved_article": "107-5450"},
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    article="107-5450",
                    brand="FORTECA",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=["LD-2"],
            )
        ],
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
    )
    preferred_miss_case = EvalCase(
        id="case-7",
        category="article_only_normalized",
        query="107 5450",
        expected_status="exact_match",
        expected_resolution_mode="article_exact",
        expected_attributes=ExpectedAttributes(
            raw_brand="FORTECA",
            resolved_brand="FORTECA",
            article="107 5450",
            resolved_article="107-5450",
        ),
        eligible_competitor_articles=["107-5451", "107-5450"],
        preferred_competitor_articles=["107-5450"],
        expected_ld_articles_by_competitor={"107-5451": ["LD-1"]},
    )
    preferred_miss_response = SimpleNamespace(
        status="exact_match",
        resolution_mode="article_exact",
        requested={"brand": "FORTECA", "article": "107 5450", "resolved_article": "107-5450"},
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    article="107-5451",
                    brand="FORTECA",
                    dn=None,
                    pn_bar=None,
                    connection=None,
                ),
                ld_articles=["LD-1"],
            )
        ],
        timing_ms={"deepseek": 1.0, "resolution": 2.0},
    )
    engine = SimpleNamespace(search_v2=lambda query, limit: good_response, attribute_extractor=None)

    good_result = _evaluate_e2e_case(engine, case, limit=5)
    hard_result = _evaluate_e2e_case(
        SimpleNamespace(
            search_v2=lambda query, limit: hard_violation_response,
            attribute_extractor=None,
        ),
        case,
        limit=5,
    )
    eligible_result = _evaluate_e2e_case(
        SimpleNamespace(
            search_v2=lambda query, limit: eligible_miss_response,
            attribute_extractor=None,
        ),
        case,
        limit=5,
    )
    ld_result = _evaluate_e2e_case(
        SimpleNamespace(
            search_v2=lambda query, limit: ld_mismatch_response,
            attribute_extractor=None,
        ),
        case,
        limit=5,
    )
    preferred_result = _evaluate_e2e_case(
        SimpleNamespace(
            search_v2=lambda query, limit: preferred_miss_response,
            attribute_extractor=None,
        ),
        preferred_miss_case,
        limit=5,
    )

    assert good_result.overall_pass is True
    assert hard_result.overall_pass is False
    assert eligible_result.overall_pass is False
    assert ld_result.overall_pass is False
    assert preferred_result.overall_pass is True
    assert preferred_result.strict_overall_pass is False


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


def test_rag_v4_category_summary_and_failure_report() -> None:
    cases = [
        RagCaseResult(
            id="case-1",
            query="Temper DN50 PN16",
            category="pn_minimum_semantics",
            expected_status="exact_match",
            actual_status="exact_match",
            resolution_mode="brand_exact",
            expected={
                "raw_brand": "Temper",
                "resolved_brand": "Temper",
                "article": None,
                "resolved_article": None,
                "dn": 50,
                "pn_bar": 16,
                "connection": None,
            },
            requested={},
            returned_competitor_articles=["a16", "a25"],
            returned_ld_articles={},
            hard_violation=False,
            eligible_hit_at_1=True,
            eligible_hit_at_5=True,
            preferred_hit_at_1=True,
            preferred_hit_at_5=True,
            ld_mapping_exact_rate=True,
            eligible_competitor_articles=["a16", "a25"],
            preferred_competitor_articles=["a16", "a25"],
            returned_top5=[],
            timing_ms={},
            comparison={"wrong_fields": [], "hallucinated_fields": [], "missing_fields": []},
        ),
        RagCaseResult(
            id="case-2",
            query="Temper DN50 PN16",
            category="pn_minimum_semantics",
            expected_status="exact_match",
            actual_status="exact_match",
            resolution_mode="brand_exact",
            expected={
                "raw_brand": "Temper",
                "resolved_brand": "Temper",
                "article": None,
                "resolved_article": None,
                "dn": 50,
                "pn_bar": 16,
                "connection": None,
            },
            requested={},
            returned_competitor_articles=["a10", "a12"],
            returned_ld_articles={},
            hard_violation=False,
            eligible_hit_at_1=False,
            eligible_hit_at_5=False,
            preferred_hit_at_1=False,
            preferred_hit_at_5=False,
            ld_mapping_exact_rate=True,
            eligible_competitor_articles=["a16", "a25", "a40"],
            preferred_competitor_articles=["a16", "a25", "a40"],
            returned_top5=[
                {
                    "rank": 1,
                    "article": "a10",
                    "brand": "Temper",
                    "dn": 50,
                    "pn_bar": 10,
                    "connection": "flanged",
                    "score": 0.42,
                    "ld_articles": ["ld-a10"],
                }
            ],
            timing_ms={},
            comparison={"wrong_fields": [], "hallucinated_fields": [], "missing_fields": []},
        ),
    ]

    by_category = _build_category_summary(cases)
    failures = _eligible_hit_5_failures(cases)

    assert by_category["pn_minimum_semantics"]["cases"] == 2
    assert by_category["pn_minimum_semantics"]["positive_cases"] == 2
    assert by_category["pn_minimum_semantics"]["eligible_hit@5"] == 0.5
    assert by_category["pn_minimum_semantics"]["overall_pass_rate"] == 0.5
    assert failures == [
        {
            "id": "case-2",
            "category": "pn_minimum_semantics",
            "query": "Temper DN50 PN16",
            "expected": {
                "raw_brand": "Temper",
                "resolved_brand": "Temper",
                "article": None,
                "resolved_article": None,
                "dn": 50,
                "pn_bar": 16,
                "connection": None,
            },
            "eligible_articles": ["a16", "a25", "a40"],
            "preferred_articles": ["a16", "a25", "a40"],
            "returned_top5": [
                {
                    "rank": 1,
                    "article": "a10",
                    "brand": "Temper",
                    "dn": 50,
                    "pn_bar": 10,
                    "connection": "flanged",
                    "score": 0.42,
                    "ld_articles": ["ld-a10"],
                }
            ],
            "resolution_mode": "brand_exact",
        }
    ]
