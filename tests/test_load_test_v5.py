from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval.load_test_v5 import (
    _build_request_schedule,
    _load_cases,
    _select_load_cases,
    _select_workload_cases,
    _stage_error_total,
)
from eval.v5_schema import EvalCase, ExpectedAttributes


def _case(case_id: str, category: str, *, expected_status: str = "exact_match") -> EvalCase:
    return EvalCase(
        id=case_id,
        category=category,
        query=f"query-{case_id}",
        expected_status=expected_status,
        expected_resolution_mode="brand_exact",
        expected_attributes=ExpectedAttributes(brand="Temper"),
    )


def test_load_cases_reads_v5_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "brand_semantic",
                "query": "Темпер DN50",
                "expected_status": "exact_match",
                "expected_resolution_mode": "brand_exact",
                "expected_attributes": {"brand": "Temper"},
                "eligible_competitor_articles": [],
                "preferred_competitor_articles": [],
                "expected_ld_articles_by_competitor": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = _load_cases(dataset)
    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].expected_resolution_mode == "brand_exact"


def test_select_load_cases_prefers_full_pipeline_cases() -> None:
    cases = [
        _case("brand-1", "brand_semantic"),
        _case("brand-2", "brand_semantic"),
        _case("typo-1", "brand_typo"),
        _case("unsupported-1", "unsupported_brand", expected_status="cannot_process"),
        _case("article-1", "article_only_exact"),
        _case("unknown-1", "unknown_article", expected_status="not_found"),
    ]

    selected = _select_load_cases(cases)
    assert [case.id for case in selected] == [
        "brand-1",
        "brand-2",
        "typo-1",
        "article-1",
        "unknown-1",
    ]


def test_select_workload_cases_supports_divided_load_groups() -> None:
    cases = [
        _case("exact-1", "brand_semantic"),
        _case("article-1", "article_only_exact"),
        _case("article-2", "article_natural_language"),
        _case("business-1", "unsupported_brand", expected_status="cannot_process"),
        _case("business-2", "dn_semantic", expected_status="cannot_process"),
        _case("mixed-1", "mixed_semantic"),
    ]

    assert {case.id for case in _select_workload_cases(cases, "full_pipeline")} == {
        "exact-1",
        "article-1",
        "article-2",
        "mixed-1",
    }
    assert {case.id for case in _select_workload_cases(cases, "article_fast_path")} == {
        "article-1",
        "article-2",
    }
    assert {case.id for case in _select_workload_cases(cases, "business_fast_fail")} == {
        "business-1",
        "business-2",
    }
    mixed = _select_workload_cases(cases, "mixed")
    assert mixed
    assert any(case.id == "exact-1" for case in mixed)
    assert any(case.id == "business-1" for case in mixed)


def test_build_request_schedule_defaults_to_full_pipeline_pool() -> None:
    cases = [
        _case("brand-1", "brand_semantic"),
        _case("brand-2", "brand_semantic"),
    ]

    schedule = _build_request_schedule(cases, 5)
    assert [case.id for case in schedule] == [
        "brand-1",
        "brand-2",
        "brand-1",
        "brand-2",
        "brand-1",
    ]


def test_stage_error_total_counts_client_timeout_once() -> None:
    stage = SimpleNamespace(
        timeout_count=0,
        upstream_error_count=0,
        client_timeout_count=1,
        connection_error_count=0,
        unexpected_client_error_count=0,
        malformed_json_count=0,
        invalid_response_count=0,
    )

    assert _stage_error_total(stage) == 1
