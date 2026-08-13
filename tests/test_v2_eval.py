from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval import evaluate_v2


def _case(
    *,
    expected_status: str = "exact_match",
    actual_status: str = "exact_match",
    eligible: list[str] | None = None,
    returned: list[str] | None = None,
    invalid: list[str] | None = None,
    gold_mode: str = "constraints",
    parser_constraint_exact: bool = True,
    raw_eligible_count: int = 1,
    filtered_eligible_count: int = 1,
    category: str = "brand_dn_pn",
    ld_mismatches: list[dict[str, object]] | None = None,
    timing_total: float = 10.0,
) -> evaluate_v2.EvaluatedCase:
    return evaluate_v2.EvaluatedCase(
        id="case",
        query="q",
        category=category,
        gold_mode=gold_mode,
        expected_constraints={},
        actual_constraints={},
        expected_status=expected_status,
        actual_status=actual_status,
        eligible_competitor_articles=eligible or ["a"],
        raw_candidate_count=3,
        raw_eligible_count=raw_eligible_count,
        filtered_candidate_count=2,
        filtered_eligible_count=filtered_eligible_count,
        returned_competitor_articles=returned or ["a"],
        invalid_returned_articles=invalid or [],
        ld_mismatches=ld_mismatches or [],
        timing_ms={"embedding": 1.0, "qdrant": 2.0, "ranking": 3.0, "total": timing_total},
        failure_stage="ok",
        raw_candidate_articles_top20=["a"],
        filtered_articles_top20=["a"],
        parser_constraint_exact=parser_constraint_exact,
    )


def test_metric_helpers_cover_hits_precision_and_invalid_rate() -> None:
    cases = [
        _case(eligible=["a"], returned=["a", "x"], invalid=["x"]),
        _case(
            eligible=["b"],
            returned=["x"],
            invalid=["x"],
            raw_eligible_count=0,
            filtered_eligible_count=0,
        ),
    ]

    assert evaluate_v2._hit_at_k(cases, 1) == pytest.approx(0.5)
    assert evaluate_v2._hit_at_k(cases, 5) == pytest.approx(0.5)
    assert evaluate_v2._precision_at_k(cases, 5) == pytest.approx(1 / 3)
    assert evaluate_v2._invalid_competitor_rate(cases) == pytest.approx(2 / 3)


def test_coverage_at_k_caps_denominator_by_k() -> None:
    cases = [
        _case(
            eligible=[str(index) for index in range(20)],
            returned=[str(index) for index in range(5)],
        )
    ]

    assert evaluate_v2._coverage_at_k(cases, 5) == pytest.approx(1.0)


def test_source_candidate_recall_uses_unique_articles() -> None:
    cases = [
        _case(
            eligible=["a", "b"],
            raw_eligible_count=2,
            filtered_eligible_count=2,
        )
    ]

    assert evaluate_v2._source_candidate_recall(cases) == pytest.approx(1.0)


def test_status_and_not_found_metrics_cover_false_exact_match() -> None:
    cases = [
        _case(
            expected_status="not_found",
            actual_status="exact_match",
            returned=["x"],
            invalid=["x"],
        ),
        _case(expected_status="not_found", actual_status="not_found", returned=[]),
        _case(expected_status="exact_match", actual_status="exact_match"),
    ]

    assert evaluate_v2._false_exact_match_rate(cases) == pytest.approx(0.5)
    assert evaluate_v2._not_found_precision(cases) == pytest.approx(1.0)
    assert evaluate_v2._not_found_recall(cases) == pytest.approx(0.5)


def test_ld_mapping_metrics_and_failure_classification() -> None:
    record = evaluate_v2.EvalRecord(
        id="case",
        query="q",
        category="brand_dn_pn",
        gold_mode="constraints",
        expected_status="exact_match",
        expected_constraints={},
        eligible_competitor_articles=["a"],
        expected_ld_articles_by_competitor={"a": ["ld1", "ld2"]},
    )
    cases = [
        _case(
            returned=["a"],
            ld_mismatches=[
                {
                    "competitor_article": "a",
                    "expected": ["ld1", "ld2"],
                    "returned": ["ld1"],
                    "hits": 1,
                    "returned_count": 1,
                }
            ],
        )
    ]

    precision, recall = evaluate_v2._ld_mapping_micro(cases, {"case": record})
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert evaluate_v2._classify_failure_stage(cases[0]) == "ld_mapping_failure"


def test_failure_stage_priority_and_percentile() -> None:
    parser_failure = _case(
        parser_constraint_exact=False,
        raw_eligible_count=1,
        filtered_eligible_count=1,
    )
    retrieval_failure = _case(raw_eligible_count=0, filtered_eligible_count=0)
    strict_filter_failure = _case(raw_eligible_count=1, filtered_eligible_count=0)
    response_failure = _case(eligible=["a"], returned=["x"], invalid=["x"])
    false_exact = _case(
        expected_status="not_found",
        actual_status="exact_match",
        returned=["x"],
        invalid=["x"],
    )

    assert evaluate_v2._classify_failure_stage(parser_failure) == "parser_failure"
    assert evaluate_v2._classify_failure_stage(retrieval_failure) == "retrieval_failure"
    assert evaluate_v2._classify_failure_stage(strict_filter_failure) == "strict_filter_failure"
    assert evaluate_v2._classify_failure_stage(response_failure) == "response_failure"
    assert evaluate_v2._classify_failure_stage(false_exact) == "false_exact_match"
    assert evaluate_v2._percentile([1.0, 3.0, 5.0], 50) == pytest.approx(3.0)
    assert evaluate_v2._percentile([1.0, 3.0, 5.0], 95) == pytest.approx(5.0)


def test_parser_only_case_does_not_call_embedder_or_qdrant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    del monkeypatch
    dataset = tmp_path / "v2_queries.jsonl"
    dataset.write_text(
        (
            '{"id":"parser_only_0001","query":"series 60","category":"parser_only",'
            '"gold_mode":"parser_only","expected_status":"parser_only",'
            '"expected_constraints":{"brand":null,"dn":null,"pn_bar":null,'
            '"connection":null,"series":"60","body_material":null},'
            '"eligible_competitor_articles":[],"expected_ld_articles_by_competitor":{}}\n'
        ),
        encoding="utf-8",
    )

    class FakeEngine:
        def __init__(self, **_: object) -> None:
            self.embedder = SimpleNamespace(
                embed_query=lambda _: (_ for _ in ()).throw(
                    RuntimeError("embedder should not be called")
                )
            )

        def readiness_status(self) -> tuple[bool, dict[str, object]]:
            return True, {"resolved_collection_name": "steel_products_active"}

        def _get_client(self) -> object:
            class FakeClient:
                def count(self, **_: object) -> object:
                    return SimpleNamespace(count=1)

            return FakeClient()

    payload = evaluate_v2.evaluate_v2(
        dataset_path=dataset,
        limit=5,
        max_cases=None,
        engine_factory=FakeEngine,
    )

    assert payload["cases"][0]["gold_mode"] == "parser_only"
    assert payload["cases"][0]["actual_status"] == "parser_only"
