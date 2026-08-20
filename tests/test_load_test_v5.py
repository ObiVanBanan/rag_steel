from __future__ import annotations

import json
from pathlib import Path

from eval.load_test_v5 import _build_request_schedule, _load_cases, _select_load_cases
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


def test_select_load_cases_builds_diverse_pool() -> None:
    cases = [
        _case("brand-1", "brand_semantic"),
        _case("brand-2", "brand_semantic"),
        _case("typo-1", "brand_typo"),
        _case("mixed-1", "mixed_semantic"),
        _case("dn-1", "dn_semantic"),
        _case("pn-1", "pn_semantic"),
        _case("connection-1", "connection_semantic"),
        _case("ambiguous-1", "ambiguous_semantic"),
        _case("unsupported-1", "unsupported_brand", expected_status="cannot_process"),
        _case("regression-1", "real_user_regression"),
        _case("article-1", "article_only_exact"),
        _case("article-2", "article_only_typo"),
        _case("brand-article-1", "brand_plus_article"),
        _case("hard-1", "article_plus_hard"),
        _case("nl-1", "article_natural_language"),
        _case("unknown-1", "unknown_article", expected_status="not_found"),
        _case("conflict-1", "brand_article_conflict", expected_status="not_found"),
    ]

    selected = _select_load_cases(cases)
    assert selected[0].id == "brand-1"
    assert {case.category for case in selected} >= {
        "brand_semantic",
        "brand_typo",
        "mixed_semantic",
        "dn_semantic",
        "pn_semantic",
        "connection_semantic",
        "ambiguous_semantic",
        "unsupported_brand",
        "real_user_regression",
        "article_only_exact",
        "article_only_typo",
        "brand_plus_article",
        "article_plus_hard",
        "article_natural_language",
        "unknown_article",
        "brand_article_conflict",
    }


def test_build_request_schedule_cycles_pool() -> None:
    cases = [
        _case("brand-1", "brand_semantic"),
        _case("unsupported-1", "unsupported_brand", expected_status="cannot_process"),
    ]

    schedule = _build_request_schedule(cases, 5)
    assert [case.id for case in schedule] == [
        "brand-1",
        "unsupported-1",
        "brand-1",
        "unsupported-1",
        "brand-1",
    ]
