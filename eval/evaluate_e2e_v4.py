"""Evaluate the full V4 end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from eval.build_v4_eval_dataset import build_v4_dataset
from eval.evaluate_rag_v4 import (
    _compare_requested,
    _dataset_sha256,
    _expected_requested,
    _extract_returned_articles,
    _git_commit,
    _hard_violation,
    _load_dataset,
    _normalize_article_id,
    _percent_hit,
    _requested_from_response,
    _safe_div,
    _status_accuracy,
)
from eval.v4_constants import DEFAULT_E2E_RESULTS_PATH, DEFAULT_GOLDEN_DATASET_PATH
from eval.v4_schema import EvalCase, ExpectedAttributes
from rag_steel.search_engine import SearchEngine
from rag_steel.settings import get_settings


@dataclass(slots=True)
class E2eCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    actual_status: str
    resolution_mode: str | None
    expected: dict[str, Any]
    requested: dict[str, Any]
    returned_competitor_articles: list[str]
    invalid_competitor_articles: list[str]
    hard_violation: bool
    eligible_hit_at_1: bool
    eligible_hit_at_5: bool
    preferred_hit_at_1: bool
    preferred_hit_at_5: bool
    overall_pass: bool
    strict_overall_pass: bool
    failure_stage: str
    timing_ms: dict[str, float]
    wall_clock_ms: float
    comparison: dict[str, list[str]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_failure_stage(
    case: EvalCase,
    *,
    actual_status: str,
    requested: ExpectedAttributes,
    hard_violation: bool,
    eligible_hit_at_5: bool,
    preferred_hit_at_5: bool,
    ld_mapping_ok: bool,
) -> str:
    expected = _expected_requested(case)
    if case.expected_status == "cannot_process":
        if actual_status != "cannot_process":
            return "status_failure"
        if requested.raw_brand is not None or requested.article is not None:
            return "resolution_failure"
        return "ok"
    if case.expected_status == "not_found":
        if actual_status != "not_found":
            return "status_failure"
        if requested.raw_brand != expected.raw_brand or requested.article != expected.article:
            return "resolution_failure"
        return "ok"
    if actual_status != case.expected_status:
        return "status_failure"
    if hard_violation:
        return "hard_filter_failure"
    if case.eligible_competitor_articles and not eligible_hit_at_5:
        return "retrieval_failure"
    if case.preferred_competitor_articles and not preferred_hit_at_5:
        return "ranking_failure"
    if not ld_mapping_ok:
        return "ld_mapping_failure"
    if requested.raw_brand != expected.raw_brand or requested.article != expected.article:
        return "resolution_failure"
    return "ok"


def _overall_pass(case: EvalCase, requested: ExpectedAttributes, response: Any) -> bool:
    if getattr(response, "status", "unknown") != case.expected_status:
        return False
    if case.expected_status in {"cannot_process", "not_found"}:
        return True
    if requested.raw_brand != _expected_requested(case).raw_brand:
        return False
    if requested.article != _expected_requested(case).article:
        return False
    return True


def _strict_overall_pass(case: EvalCase, requested: ExpectedAttributes, response: Any) -> bool:
    if not _overall_pass(case, requested, response):
        return False
    if case.preferred_competitor_articles and not _percent_hit(
        _extract_returned_articles(response),
        case.preferred_competitor_articles,
        5,
    ):
        return False
    return True


def _ld_mapping_ok(case: EvalCase, response: Any) -> bool:
    by_article: dict[str, list[str]] = {}
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if not article:
            continue
        by_article[_normalize_article_id(article)] = [
            _normalize_article_id(item) for item in getattr(result, "ld_articles", []) or []
        ]
    if not by_article:
        return True
    for article, _actual in by_article.items():
        if article not in case.expected_ld_articles_by_competitor:
            continue
        expected = case.expected_ld_articles_by_competitor[article]
        if set(by_article.get(article, [])) != set(
            _normalize_article_id(item) for item in expected
        ):
            return False
    return True


def _evaluate_case(engine: SearchEngine, case: EvalCase, *, limit: int) -> E2eCaseResult:
    started = perf_counter()
    try:
        response = engine.search_v2(case.query, limit=limit)
        wall_clock_ms = (perf_counter() - started) * 1000.0
        actual_requested = _requested_from_response(response)
        returned_competitor_articles = _extract_returned_articles(response)
        invalid = [
            article
            for article in returned_competitor_articles
            if article not in case.eligible_competitor_articles
        ]
        hard_violation = _hard_violation(case.expected_attributes, response)
        eligible_hit_at_1 = _percent_hit(
            returned_competitor_articles, case.eligible_competitor_articles, 1
        )
        eligible_hit_at_5 = _percent_hit(
            returned_competitor_articles, case.eligible_competitor_articles, 5
        )
        preferred_hit_at_1 = _percent_hit(
            returned_competitor_articles, case.preferred_competitor_articles, 1
        )
        preferred_hit_at_5 = _percent_hit(
            returned_competitor_articles, case.preferred_competitor_articles, 5
        )
        ld_mapping_ok = _ld_mapping_ok(case, response)
        overall_pass = _overall_pass(case, actual_requested, response)
        strict_overall_pass = _strict_overall_pass(case, actual_requested, response)
        failure_stage = _classify_failure_stage(
            case,
            actual_status=getattr(response, "status", "unknown"),
            requested=actual_requested,
            hard_violation=hard_violation,
            eligible_hit_at_5=eligible_hit_at_5,
            preferred_hit_at_5=preferred_hit_at_5,
            ld_mapping_ok=ld_mapping_ok,
        )
        comparison = _compare_requested(_expected_requested(case), actual_requested)
        return E2eCaseResult(
            id=case.id,
            query=case.query,
            category=case.category,
            expected_status=case.expected_status,
            actual_status=getattr(response, "status", "unknown"),
            resolution_mode=getattr(response, "resolution_mode", None),
            expected=case.expected_attributes.model_dump(mode="json"),
            requested=actual_requested.model_dump(mode="json"),
            returned_competitor_articles=returned_competitor_articles,
            invalid_competitor_articles=invalid,
            hard_violation=hard_violation,
            eligible_hit_at_1=eligible_hit_at_1,
            eligible_hit_at_5=eligible_hit_at_5,
            preferred_hit_at_1=preferred_hit_at_1,
            preferred_hit_at_5=preferred_hit_at_5,
            overall_pass=overall_pass,
            strict_overall_pass=strict_overall_pass,
            failure_stage=failure_stage,
            timing_ms=dict(getattr(response, "timing_ms", {}) or {}),
            wall_clock_ms=wall_clock_ms,
            comparison=comparison,
        )
    except Exception as exc:
        wall_clock_ms = (perf_counter() - started) * 1000.0
        return E2eCaseResult(
            id=case.id,
            query=case.query,
            category=case.category,
            expected_status=case.expected_status,
            actual_status="technical_failure",
            resolution_mode=None,
            expected=case.expected_attributes.model_dump(mode="json"),
            requested=ExpectedAttributes().model_dump(mode="json"),
            returned_competitor_articles=[],
            invalid_competitor_articles=[],
            hard_violation=False,
            eligible_hit_at_1=False,
            eligible_hit_at_5=False,
            preferred_hit_at_1=False,
            preferred_hit_at_5=False,
            overall_pass=False,
            strict_overall_pass=False,
            failure_stage="technical_failure",
            timing_ms={},
            wall_clock_ms=wall_clock_ms,
            comparison={"wrong_fields": [], "hallucinated_fields": [], "missing_fields": []},
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_e2e_v4(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    limit: int = 5,
    engine_factory: Callable[..., SearchEngine] = SearchEngine,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v4_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    settings = get_settings()
    engine = engine_factory()

    cases = [_evaluate_case(engine, case, limit=limit) for case in dataset]
    positive_cases = [case for case in cases if case.expected_status == "exact_match"]
    summary = {
        "cases": len(cases),
        "status_accuracy": _status_accuracy(cases),
        "e2e_preferred_hit@1": _safe_div(
            sum(1 for case in positive_cases if case.preferred_hit_at_1), len(positive_cases)
        ),
        "e2e_preferred_hit@5": _safe_div(
            sum(1 for case in positive_cases if case.preferred_hit_at_5), len(positive_cases)
        ),
        "e2e_eligible_hit@1": _safe_div(
            sum(1 for case in positive_cases if case.eligible_hit_at_1), len(positive_cases)
        ),
        "e2e_eligible_hit@5": _safe_div(
            sum(1 for case in positive_cases if case.eligible_hit_at_5), len(positive_cases)
        ),
        "overall_pass_rate": _safe_div(sum(1 for case in cases if case.overall_pass), len(cases)),
        "strict_overall_pass_rate": _safe_div(
            sum(1 for case in cases if case.strict_overall_pass), len(cases)
        ),
    }

    grouped: dict[str, list[E2eCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.resolution_mode or "unknown"].append(case)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "deepseek_model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "qdrant_alias": settings.qdrant_collection_alias,
        "limit": limit,
        "summary": summary,
        "by_resolution_mode": {mode: len(items) for mode, items in sorted(grouped.items())},
        "failure_counts": dict(Counter(case.failure_stage for case in cases)),
        "cases": [case.to_dict() for case in cases],
    }
    return payload


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def render_report(payload: dict[str, Any], output_path: Path) -> str:
    summary = payload["summary"]
    lines = [
        "# E2E V4 Evaluation",
        "",
        "## Overall",
        f"- cases: `{summary['cases']}`",
        f"- status accuracy: `{summary['status_accuracy']:.4f}`",
        (
            f"- e2e preferred hit@1/@5: "
            f"`{summary['e2e_preferred_hit@1']:.4f}` / "
            f"`{summary['e2e_preferred_hit@5']:.4f}`"
        ),
        (
            f"- e2e eligible hit@1/@5: "
            f"`{summary['e2e_eligible_hit@1']:.4f}` / "
            f"`{summary['e2e_eligible_hit@5']:.4f}`"
        ),
        f"- overall pass rate: `{summary['overall_pass_rate']:.4f}`",
        f"- strict overall pass rate: `{summary['strict_overall_pass_rate']:.4f}`",
        "",
        "## By Resolution Mode",
    ]
    for mode, count in sorted(payload["by_resolution_mode"].items()):
        lines.append(f"- {mode}: `{count}`")
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate E2E for V4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_E2E_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/e2e_v4_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_e2e_v4(dataset_path=args.dataset, max_cases=args.max_cases)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
