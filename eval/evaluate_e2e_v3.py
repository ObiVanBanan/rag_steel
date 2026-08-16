"""Evaluate the full production end-to-end pipeline for V3."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from rag_steel.search_engine import SearchEngine
from rag_steel.settings import get_settings

from eval.build_v3_eval_dataset import build_v3_dataset
from eval.v3_common import (
    _normalize_article_id,
    _percentile,
    _safe_div,
    compare_expected_actual,
    hard_exact_match,
)
from eval.v3_constants import DEFAULT_E2E_RESULTS_PATH, DEFAULT_GOLDEN_DATASET_PATH
from eval.v3_schema import EvalCase, ExpectedAttributes


@dataclass(slots=True)
class E2eCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    actual_status: str
    expected: dict[str, Any]
    actual: dict[str, Any]
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(EvalCase.model_validate_json(line))
    return cases


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _dataset_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _extract_returned_articles(response: Any) -> list[str]:
    articles: list[str] = []
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if article:
            normalized = _normalize_article_id(article)
            if normalized not in articles:
                articles.append(normalized)
    return articles


def _hard_violation(expected: ExpectedAttributes, response: Any) -> bool:
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        if competitor is None:
            continue
        competitor_attrs = ExpectedAttributes.model_validate(
            {
                "brand": getattr(competitor, "brand", None),
                "dn": getattr(competitor, "dn", None),
                "pn_bar": getattr(competitor, "pn_bar", None),
                "connection": getattr(competitor, "connection", None),
            }
        )
        if not hard_exact_match(expected, competitor_attrs):
            return True
    return False


def _ld_mapping_ok(case: EvalCase, response: Any) -> bool:
    if not case.expected_ld_articles_by_competitor:
        return True
    by_article: dict[str, list[str]] = {}
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if not article:
            continue
        by_article[_normalize_article_id(article)] = [
            _normalize_article_id(item) for item in getattr(result, "ld_articles", []) or []
        ]
    for article, expected in case.expected_ld_articles_by_competitor.items():
        if set(by_article.get(article, [])) != set(_normalize_article_id(item) for item in expected):
            return False
    return True


def _rank_metrics(returned: list[str], target: list[str]) -> tuple[bool, bool, bool, float]:
    target_set = set(target)
    hit1 = bool(set(returned[:1]) & target_set)
    hit5 = bool(set(returned[:5]) & target_set)
    precision5 = _safe_div(sum(1 for article in returned[:5] if article in target_set), len(returned[:5]))
    mrr = 0.0
    for index, article in enumerate(returned, start=1):
        if article in target_set:
            mrr = 1.0 / index
            break
    return hit1, hit5, hit5, precision5


def _classify_failure_stage(
    case: EvalCase,
    *,
    response: Any,
    actual_requested: ExpectedAttributes,
    hard_violation: bool,
    eligible_hit_at_5: bool,
    preferred_hit_at_5: bool,
    ld_mapping_ok: bool,
) -> str:
    if case.expected_status != getattr(response, "status", "unknown"):
        return "status_failure"
    if case.expected_status == "cannot_process":
        return "ok"
    if case.expected_attributes.brand != actual_requested.brand:
        return "brand_gate_failure"
    if not hard_exact_match(case.expected_attributes, actual_requested):
        return "deepseek_failure"
    if hard_violation:
        return "hard_filter_failure"
    if case.eligible_competitor_articles and not eligible_hit_at_5:
        return "retrieval_failure"
    if case.preferred_competitor_articles and not preferred_hit_at_5:
        return "ranking_failure"
    if not ld_mapping_ok:
        return "ld_mapping_failure"
    return "ok"


def _evaluate_case(engine: SearchEngine, case: EvalCase) -> E2eCaseResult:
    started = perf_counter()
    try:
        response = engine.search_v2(case.query, limit=5)
        wall_clock_ms = (perf_counter() - started) * 1000.0
        requested = ExpectedAttributes.model_validate(response.requested or {})
        returned_competitor_articles = _extract_returned_articles(response)
        invalid_competitor_articles = [
            article for article in returned_competitor_articles if article not in case.eligible_competitor_articles
        ]
        hard_violation = _hard_violation(case.expected_attributes, response)
        eligible_hit_at_1 = bool(set(returned_competitor_articles[:1]) & set(case.eligible_competitor_articles))
        eligible_hit_at_5 = bool(set(returned_competitor_articles[:5]) & set(case.eligible_competitor_articles))
        preferred_hit_at_1 = bool(set(returned_competitor_articles[:1]) & set(case.preferred_competitor_articles))
        preferred_hit_at_5 = bool(set(returned_competitor_articles[:5]) & set(case.preferred_competitor_articles))
        overall_pass = (
            case.expected_status == getattr(response, "status", "unknown")
            and (
                case.expected_status in {"cannot_process", "not_found"}
                or (
                    hard_exact_match(case.expected_attributes, requested)
                    and not hard_violation
                    and eligible_hit_at_5
                    and _ld_mapping_ok(case, response)
                )
            )
        )
        strict_overall_pass = overall_pass and (
            not case.preferred_competitor_articles or preferred_hit_at_5
        )
        failure_stage = _classify_failure_stage(
            case,
            response=response,
            actual_requested=requested,
            hard_violation=hard_violation,
            eligible_hit_at_5=eligible_hit_at_5,
            preferred_hit_at_5=preferred_hit_at_5,
            ld_mapping_ok=_ld_mapping_ok(case, response),
        )
        comparison = compare_expected_actual(case.expected_attributes, requested)
        return E2eCaseResult(
            id=case.id,
            query=case.query,
            category=case.category,
            expected_status=case.expected_status,
            actual_status=getattr(response, "status", "unknown"),
            expected=case.expected_attributes.model_dump(mode="json"),
            actual=requested.model_dump(mode="json"),
            returned_competitor_articles=returned_competitor_articles,
            invalid_competitor_articles=invalid_competitor_articles,
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
    except Exception:
        wall_clock_ms = (perf_counter() - started) * 1000.0
        return E2eCaseResult(
            id=case.id,
            query=case.query,
            category=case.category,
            expected_status=case.expected_status,
            actual_status="technical_failure",
            expected=case.expected_attributes.model_dump(mode="json"),
            actual=ExpectedAttributes().model_dump(mode="json"),
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
        )


def _status_accuracy(cases: list[E2eCaseResult]) -> float:
    if not cases:
        return 0.0
    return _safe_div(sum(1 for case in cases if case.expected_status == case.actual_status), len(cases))


def _precision_recall(cases: list[E2eCaseResult], status: str) -> tuple[float, float]:
    predicted = [case for case in cases if case.actual_status == status]
    expected = [case for case in cases if case.expected_status == status]
    precision = _safe_div(sum(1 for case in predicted if case.expected_status == status), len(predicted))
    recall = _safe_div(sum(1 for case in expected if case.actual_status == status), len(expected))
    return precision, recall


def _latency_summary(cases: list[E2eCaseResult]) -> dict[str, float]:
    wall_clock = [case.wall_clock_ms for case in cases]
    embedding = [case.timing_ms.get("embedding", 0.0) for case in cases]
    qdrant = [case.timing_ms.get("qdrant", 0.0) for case in cases]
    ranking = [case.timing_ms.get("ranking", 0.0) for case in cases]
    return {
        "wall_clock_p50_ms": _percentile(wall_clock, 50),
        "wall_clock_p95_ms": _percentile(wall_clock, 95),
        "wall_clock_p99_ms": _percentile(wall_clock, 99),
        "embedding_p50_ms": _percentile(embedding, 50),
        "embedding_p95_ms": _percentile(embedding, 95),
        "qdrant_p50_ms": _percentile(qdrant, 50),
        "qdrant_p95_ms": _percentile(qdrant, 95),
        "ranking_p50_ms": _percentile(ranking, 50),
        "ranking_p95_ms": _percentile(ranking, 95),
    }


def evaluate_e2e_v3(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    limit: int = 5,
    engine_factory: Callable[..., SearchEngine] = SearchEngine,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v3_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    settings = get_settings()
    engine = engine_factory()
    if hasattr(engine, "readiness_status"):
        ready, readiness = engine.readiness_status()
        if not ready:
            raise RuntimeError(json.dumps(readiness, ensure_ascii=False))
    else:
        readiness = {"resolved_collection_name": None}

    cases = [_evaluate_case(engine, case) for case in dataset]
    overall_pass_rate = _safe_div(sum(1 for case in cases if case.overall_pass), len(cases))
    strict_overall_pass_rate = _safe_div(
        sum(1 for case in cases if case.strict_overall_pass), len(cases)
    )
    cannot_process_precision, cannot_process_recall = _precision_recall(cases, "cannot_process")
    not_found_precision, not_found_recall = _precision_recall(cases, "not_found")
    eligible_hit_at_1 = _safe_div(sum(1 for case in cases if case.eligible_hit_at_1), len(cases))
    eligible_hit_at_5 = _safe_div(sum(1 for case in cases if case.eligible_hit_at_5), len(cases))
    preferred_hit_at_1 = _safe_div(sum(1 for case in cases if case.preferred_hit_at_1), len(cases))
    preferred_hit_at_5 = _safe_div(sum(1 for case in cases if case.preferred_hit_at_5), len(cases))

    summary = {
        "cases": len(cases),
        "status_accuracy": _status_accuracy(cases),
        "cannot_process_precision": cannot_process_precision,
        "cannot_process_recall": cannot_process_recall,
        "not_found_precision": not_found_precision,
        "not_found_recall": not_found_recall,
        "e2e_preferred_hit@1": preferred_hit_at_1,
        "e2e_preferred_hit@5": preferred_hit_at_5,
        "e2e_eligible_hit@1": eligible_hit_at_1,
        "e2e_eligible_hit@5": eligible_hit_at_5,
        "invalid_competitor_rate": _safe_div(
            sum(len(case.invalid_competitor_articles) for case in cases),
            sum(len(case.returned_competitor_articles) for case in cases) or 1,
        ),
        "overall_pass_rate": overall_pass_rate,
        "strict_overall_pass_rate": strict_overall_pass_rate,
        **_latency_summary(cases),
    }

    grouped: dict[str, list[E2eCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

    failure_counts = Counter(case.failure_stage for case in cases)
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
        "resolved_collection": readiness.get("resolved_collection_name"),
        "limit": limit,
        "source_candidate_limit": settings.source_candidate_limit,
        "timestamp": run_id,
        "summary": summary,
        "by_category": {
            category: {
                "cases": len(items),
                "status_accuracy": _status_accuracy(items),
                "overall_pass_rate": _safe_div(
                    sum(1 for item in items if item.overall_pass), len(items)
                ),
                "strict_overall_pass_rate": _safe_div(
                    sum(1 for item in items if item.strict_overall_pass), len(items)
                ),
            }
            for category, items in sorted(grouped.items())
        },
        "failure_counts": dict(failure_counts),
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
        "# E2E V3 Evaluation",
        "",
        "## Run",
        f"- commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- dataset: `{payload['dataset_path']}`",
        f"- cases: `{summary['cases']}`",
        f"- collection: `{payload.get('resolved_collection') or 'unknown'}`",
        "",
        "## Overall",
        f"- status accuracy: `{summary['status_accuracy']:.4f}`",
        f"- cannot_process precision/recall: `{summary['cannot_process_precision']:.4f}` / `{summary['cannot_process_recall']:.4f}`",
        f"- not_found precision/recall: `{summary['not_found_precision']:.4f}` / `{summary['not_found_recall']:.4f}`",
        f"- e2e preferred hit@1/@5: `{summary['e2e_preferred_hit@1']:.4f}` / `{summary['e2e_preferred_hit@5']:.4f}`",
        f"- e2e eligible hit@1/@5: `{summary['e2e_eligible_hit@1']:.4f}` / `{summary['e2e_eligible_hit@5']:.4f}`",
        f"- invalid competitor rate: `{summary['invalid_competitor_rate']:.4f}`",
        f"- overall pass rate: `{summary['overall_pass_rate']:.4f}`",
        f"- strict overall pass rate: `{summary['strict_overall_pass_rate']:.4f}`",
        f"- wall-clock p50/p95/p99: `{summary['wall_clock_p50_ms']:.1f}` / `{summary['wall_clock_p95_ms']:.1f}` / `{summary['wall_clock_p99_ms']:.1f}` ms",
        f"- embedding p50/p95: `{summary['embedding_p50_ms']:.1f}` / `{summary['embedding_p95_ms']:.1f}` ms",
        f"- qdrant p50/p95: `{summary['qdrant_p50_ms']:.1f}` / `{summary['qdrant_p95_ms']:.1f}` ms",
        f"- ranking p50/p95: `{summary['ranking_p50_ms']:.1f}` / `{summary['ranking_p95_ms']:.1f}` ms",
        "",
        "## By Category",
        "",
        "| Category | Cases | Status Acc | Overall | Strict |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, metrics in sorted(payload["by_category"].items()):
        lines.append(
            f"| {category} | {metrics['cases']} | {metrics['status_accuracy']:.4f} | {metrics['overall_pass_rate']:.4f} | {metrics['strict_overall_pass_rate']:.4f} |"
        )

    lines.extend(["", "## Failure Stages", ""])
    for stage, count in sorted(payload["failure_counts"].items()):
        lines.append(f"- {stage}: {count}")

    grouped_failures = defaultdict(list)
    for case in payload["cases"]:
        if case["failure_stage"] != "ok":
            grouped_failures[case["failure_stage"]].append(case)

    lines.extend(["", "## Worst Failures", ""])
    for stage in (
        "technical_failure",
        "brand_gate_failure",
        "deepseek_failure",
        "hard_filter_failure",
        "retrieval_failure",
        "ranking_failure",
        "ld_mapping_failure",
        "status_failure",
    ):
        stage_cases = grouped_failures.get(stage, [])[:20]
        if not stage_cases:
            continue
        lines.append(f"### {stage}")
        for case in stage_cases:
            lines.append(
                f"- `{case['id']}` `{case['query']}` expected `{case['expected_status']}` actual `{case['actual_status']}`"
            )
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the production V3 pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_E2E_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/e2e_v3_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_e2e_v3(dataset_path=args.dataset, max_cases=args.max_cases, limit=args.limit)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    print(str(args.output_json))
    return 0


__all__ = [
    "E2eCaseResult",
    "evaluate_e2e_v3",
    "render_report",
    "write_results_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
