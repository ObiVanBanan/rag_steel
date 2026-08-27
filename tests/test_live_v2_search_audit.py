from __future__ import annotations

from eval.live_v2_search_audit import (
    AuditRecord,
    article_identity_equal,
    check_response,
    expected_reason_matches,
    expected_status_matches,
    has_duplicate_ld_articles,
    metric_or_none,
    pn_meets_minimum,
    summarize,
)


def _response(
    *,
    status: str = "exact_match",
    reason_code: str | None = None,
    requested: dict[str, object] | None = None,
    results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "requested": requested or {},
        "results": results or [],
        "timing_ms": {},
        "_client_total_ms": 1.0,
    }
    if reason_code is not None:
        payload["reason"] = {"code": reason_code}
    return payload


def _result(
    *,
    article: str = "A-1",
    brand: str = "Temper",
    dn: float = 50,
    pn_bar: float = 40,
    connection: str = "фланцевое",
    body_material: str = "сталь 20",
    ld_articles: list[str] | None = None,
) -> dict[str, object]:
    return {
        "competitor": {
            "article": article,
            "brand": brand,
            "dn": dn,
            "pn_bar": pn_bar,
            "connection": connection,
            "body_material": body_material,
        },
        "ld_articles": ld_articles or ["LD-1", "LD-2"],
    }


def test_article_identity_uses_compact_normalization() -> None:
    assert article_identity_equal("КШ.Ф.П.Р.015.40-01", "КШ.ФПР.015.40-01")


def test_pn_minimum_checker() -> None:
    assert pn_meets_minimum(40, 16)
    assert pn_meets_minimum("25", 25)
    assert not pn_meets_minimum(16, 25)


def test_duplicate_ld_detector() -> None:
    assert has_duplicate_ld_articles({"ld_articles": ["LD-1", "LD.1"]})
    assert not has_duplicate_ld_articles({"ld_articles": ["LD-1", "LD-2"]})


def test_expected_status_and_reason_checkers() -> None:
    assert expected_status_matches(
        AuditRecord(id="a", category="c", query="q", expected_status="x"), "x"
    )
    assert expected_status_matches(
        AuditRecord(id="a", category="c", query="q", expected_statuses=["x", "y"]),
        "y",
    )
    assert expected_reason_matches(
        AuditRecord(id="a", category="c", query="q", expected_reason_code="ARTICLE_NOT_FOUND"),
        {"reason": {"code": "ARTICLE_NOT_FOUND"}},
    )


def test_missing_timing_or_metric_denominator_returns_none() -> None:
    assert metric_or_none(0, 0) is None
    case = check_response(
        AuditRecord(id="a", category="no_metric", query="q", expected_status="cannot_process"),
        200,
        _response(status="cannot_process"),
    )

    summary = summarize([case])

    assert summary["extraction_dn_accuracy"] is None
    assert summary["article_identity_accuracy"] is None


def test_failure_classification_product_bug() -> None:
    case = check_response(
        AuditRecord(
            id="conflict",
            category="connection_conflict",
            query="Temper DN50 фланцевое резьбовое",
            expected_statuses=["cannot_process", "not_found"],
            expected_failure_code="BUG_CONNECTION_CONFLICT_NOT_DETECTED",
            expected_classification="PRODUCT_BUG",
        ),
        200,
        _response(status="exact_match", results=[_result()]),
    )

    assert case.classification == "PRODUCT_BUG"
    assert "BUG_CONNECTION_CONFLICT_NOT_DETECTED" in case.issues
    assert case.metrics["conflict_detection"] is False


def test_normalized_article_alias_classifies_checker_issue() -> None:
    case = check_response(
        AuditRecord(
            id="alias",
            category="article_identity",
            query="КШ.Ф.П.Р.015.40-01",
            expected_status="exact_match",
            expected_article="КШ.Ф.П.Р.015.40-01",
            expected_classification="EVAL_CHECKER_BUG",
        ),
        200,
        _response(
            requested={"article": "КШ.Ф.П.Р.015.40-01"},
            results=[_result(article="КШ.ФПР.015.40-01")],
        ),
    )

    assert case.classification == "EVAL_CHECKER_BUG"
    assert case.metrics["article_identity"] is True


def test_hard_constraints_check_all_results_with_pn_minimum() -> None:
    case = check_response(
        AuditRecord(
            id="hard",
            category="steel",
            query="Temper DN50 PN16 фланцевое",
            expected_status="exact_match",
            expected_brand="Temper",
            expected_dn=50,
            expected_pn_min=16,
            expected_connection="фланцевое",
        ),
        200,
        _response(results=[_result(pn_bar=40), _result(pn_bar=25)]),
    )

    assert case.classification == "PASS"
    assert case.metrics["hard_constraints"] is True
    assert case.metrics["no_duplicate_results"] is True
