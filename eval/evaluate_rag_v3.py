"""Evaluate the production search pipeline with gold extractor inputs for V3."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from rag_steel.search_engine import SearchEngine
from rag_steel.settings import get_settings

from eval.build_v3_eval_dataset import build_v3_dataset
from eval.v3_common import (
    GoldAttributeExtractor,
    GoldBrandDetector,
    _normalize_article_id,
    _percentile,
    _safe_div,
    compare_expected_actual,
    document_article,
    hard_exact_match,
)
from eval.v3_constants import DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_RAG_RESULTS_PATH
from eval.v3_schema import EvalCase, ExpectedAttributes


@dataclass(slots=True)
class RagCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    actual_status: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    eligible_competitor_articles: list[str]
    preferred_competitor_articles: list[str]
    returned_competitor_articles: list[str]
    returned_ld_articles: dict[str, list[str]]
    invalid_competitor_articles: list[str]
    hard_violation: bool
    preferred_hit_at_1: bool
    preferred_hit_at_3: bool
    preferred_hit_at_5: bool
    eligible_hit_at_1: bool
    eligible_hit_at_5: bool
    preferred_precision_at_5: float
    mrr: float
    eligible_coverage_at_5: float
    ld_mapping_precision: float
    ld_mapping_recall: float
    ld_mapping_exact_rate: bool
    timing_ms: dict[str, float]
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


def _extract_returned_ld_articles(response: Any) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if not article:
            continue
        normalized_article = _normalize_article_id(article)
        ld_articles = []
        for ld_article in getattr(result, "ld_articles", []) or []:
            normalized_ld = _normalize_article_id(ld_article)
            if normalized_ld not in ld_articles:
                ld_articles.append(normalized_ld)
        mapping[normalized_article] = ld_articles
    return mapping


def _percent_hit(returned: list[str], target: list[str], k: int) -> bool:
    return bool(set(returned[:k]) & set(target))


def _precision_at_k(returned: list[str], target: list[str], k: int) -> float:
    top = returned[:k]
    if not top:
        return 0.0
    return _safe_div(sum(1 for article in top if article in target), len(top))


def _mrr(returned: list[str], target: list[str]) -> float:
    target_set = set(target)
    for index, article in enumerate(returned, start=1):
        if article in target_set:
            return 1.0 / index
    return 0.0


def _coverage_at_k(returned: list[str], target: list[str], k: int) -> float:
    if not target:
        return 0.0
    return _safe_div(len(set(returned[:k]) & set(target)), len(set(target)))


def _ld_metrics(case: EvalCase, returned: dict[str, list[str]]) -> tuple[float, float, bool]:
    if not case.expected_ld_articles_by_competitor:
        return 0.0, 0.0, True
    precisions: list[float] = []
    recalls: list[float] = []
    exact = True
    for article, expected_ld in case.expected_ld_articles_by_competitor.items():
        actual_ld = returned.get(article, [])
        precisions.append(_safe_div(len(set(actual_ld) & set(expected_ld)), len(actual_ld)))
        recalls.append(_safe_div(len(set(actual_ld) & set(expected_ld)), len(expected_ld)))
        if set(actual_ld) != set(expected_ld):
            exact = False
    return (
        _safe_div(sum(precisions), len(precisions)),
        _safe_div(sum(recalls), len(recalls)),
        exact,
    )


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


def _evaluate_case(engine: SearchEngine, case: EvalCase) -> RagCaseResult:
    if case.expected_status == "cannot_process":
        response = engine.search_v2(case.query, limit=5)
    else:
        response = engine.search_v2(case.query, limit=5)

    returned_articles = _extract_returned_articles(response)
    returned_ld_articles = _extract_returned_ld_articles(response)
    invalid = [article for article in returned_articles if article not in case.eligible_competitor_articles]
    comparison = compare_expected_actual(
        case.expected_attributes,
        ExpectedAttributes.model_validate(response.requested or {}),
    )
    eligible_target = case.eligible_competitor_articles
    preferred_target = case.preferred_competitor_articles

    ld_precision, ld_recall, ld_exact = _ld_metrics(case, returned_ld_articles)

    return RagCaseResult(
        id=case.id,
        query=case.query,
        category=case.category,
        expected_status=case.expected_status,
        actual_status=getattr(response, "status", "unknown"),
        expected=case.expected_attributes.model_dump(mode="json"),
        actual=ExpectedAttributes.model_validate(response.requested or {}).model_dump(mode="json"),
        eligible_competitor_articles=eligible_target,
        preferred_competitor_articles=preferred_target,
        returned_competitor_articles=returned_articles,
        returned_ld_articles=returned_ld_articles,
        invalid_competitor_articles=invalid,
        hard_violation=_hard_violation(case.expected_attributes, response),
        preferred_hit_at_1=_percent_hit(returned_articles, preferred_target, 1),
        preferred_hit_at_3=_percent_hit(returned_articles, preferred_target, 3),
        preferred_hit_at_5=_percent_hit(returned_articles, preferred_target, 5),
        eligible_hit_at_1=_percent_hit(returned_articles, eligible_target, 1),
        eligible_hit_at_5=_percent_hit(returned_articles, eligible_target, 5),
        preferred_precision_at_5=_precision_at_k(returned_articles, preferred_target, 5),
        mrr=_mrr(returned_articles, preferred_target),
        eligible_coverage_at_5=_coverage_at_k(returned_articles, eligible_target, 5),
        ld_mapping_precision=ld_precision,
        ld_mapping_recall=ld_recall,
        ld_mapping_exact_rate=ld_exact,
        timing_ms=dict(getattr(response, "timing_ms", {}) or {}),
        comparison=comparison,
    )


def _overall_pass(case: RagCaseResult) -> bool:
    if case.expected_status != case.actual_status:
        return False
    if case.expected_status in {"cannot_process", "not_found"}:
        return True
    if case.hard_violation:
        return False
    if not case.eligible_hit_at_5:
        return False
    if not case.ld_mapping_exact_rate:
        return False
    return True


def _strict_overall_pass(case: RagCaseResult) -> bool:
    if not _overall_pass(case):
        return False
    if case.preferred_competitor_articles and not case.preferred_hit_at_5:
        return False
    return True


def _status_accuracy(cases: list[RagCaseResult]) -> float:
    if not cases:
        return 0.0
    return _safe_div(sum(1 for case in cases if case.expected_status == case.actual_status), len(cases))


def _not_found_precision(cases: list[RagCaseResult]) -> float:
    predicted = [case for case in cases if case.actual_status == "not_found"]
    if not predicted:
        return 0.0
    return _safe_div(sum(1 for case in predicted if case.expected_status == "not_found"), len(predicted))


def _not_found_recall(cases: list[RagCaseResult]) -> float:
    expected = [case for case in cases if case.expected_status == "not_found"]
    if not expected:
        return 0.0
    return _safe_div(sum(1 for case in expected if case.actual_status == "not_found"), len(expected))


def _cannot_process_precision(cases: list[RagCaseResult]) -> float:
    predicted = [case for case in cases if case.actual_status == "cannot_process"]
    if not predicted:
        return 0.0
    return _safe_div(
        sum(1 for case in predicted if case.expected_status == "cannot_process"),
        len(predicted),
    )


def _cannot_process_recall(cases: list[RagCaseResult]) -> float:
    expected = [case for case in cases if case.expected_status == "cannot_process"]
    if not expected:
        return 0.0
    return _safe_div(
        sum(1 for case in expected if case.actual_status == "cannot_process"),
        len(expected),
    )


def _invalid_competitor_rate(cases: list[RagCaseResult]) -> float:
    total_returned = sum(len(case.returned_competitor_articles) for case in cases)
    if total_returned == 0:
        return 0.0
    invalid = sum(len(case.invalid_competitor_articles) for case in cases)
    return _safe_div(invalid, total_returned)


def _latency_summary(cases: list[RagCaseResult]) -> dict[str, float]:
    embedding = [case.timing_ms.get("embedding", 0.0) for case in cases]
    qdrant = [case.timing_ms.get("qdrant", 0.0) for case in cases]
    ranking = [case.timing_ms.get("ranking", 0.0) for case in cases]
    return {
        "embedding_p50_ms": _percentile(embedding, 50),
        "embedding_p95_ms": _percentile(embedding, 95),
        "qdrant_p50_ms": _percentile(qdrant, 50),
        "qdrant_p95_ms": _percentile(qdrant, 95),
        "ranking_p50_ms": _percentile(ranking, 50),
        "ranking_p95_ms": _percentile(ranking, 95),
    }


def evaluate_rag_v3(
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
    engine = engine_factory(
        brand_detector=GoldBrandDetector(None),
        attribute_extractor=GoldAttributeExtractor(ExpectedAttributes()),
    )
    if hasattr(engine, "readiness_status"):
        ready, readiness = engine.readiness_status()
        if not ready:
            raise RuntimeError(json.dumps(readiness, ensure_ascii=False))
    else:
        readiness = {"resolved_collection_name": None}

    cases = []
    for case in dataset:
        engine.brand_detector = GoldBrandDetector(case.expected_attributes.brand)
        engine.attribute_extractor = GoldAttributeExtractor(case.expected_attributes)
        cases.append(_evaluate_case(engine, case))

    summary = {
        "cases": len(cases),
        "status_accuracy": _status_accuracy(cases),
        "cannot_process_precision": _cannot_process_precision(cases),
        "cannot_process_recall": _cannot_process_recall(cases),
        "not_found_precision": _not_found_precision(cases),
        "not_found_recall": _not_found_recall(cases),
        "hard_violation_rate": _safe_div(sum(1 for case in cases if case.hard_violation), len(cases)),
        "preferred_hit@1": _safe_div(sum(1 for case in cases if case.preferred_hit_at_1), len(cases)),
        "preferred_hit@3": _safe_div(sum(1 for case in cases if case.preferred_hit_at_3), len(cases)),
        "preferred_hit@5": _safe_div(sum(1 for case in cases if case.preferred_hit_at_5), len(cases)),
        "preferred_precision@5": _safe_div(
            sum(case.preferred_precision_at_5 for case in cases), len(cases)
        ),
        "MRR": _safe_div(sum(case.mrr for case in cases), len(cases)),
        "eligible_hit@1": _safe_div(sum(1 for case in cases if case.eligible_hit_at_1), len(cases)),
        "eligible_hit@5": _safe_div(sum(1 for case in cases if case.eligible_hit_at_5), len(cases)),
        "eligible_coverage@5": _safe_div(
            sum(case.eligible_coverage_at_5 for case in cases), len(cases)
        ),
        "ld_mapping_precision": _safe_div(
            sum(case.ld_mapping_precision for case in cases), len(cases)
        ),
        "ld_mapping_recall": _safe_div(sum(case.ld_mapping_recall for case in cases), len(cases)),
        "ld_mapping_exact_rate": _safe_div(
            sum(1 for case in cases if case.ld_mapping_exact_rate), len(cases)
        ),
        "invalid_competitor_rate": _invalid_competitor_rate(cases),
        **_latency_summary(cases),
    }

    grouped: dict[str, list[RagCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

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
                "hard_violation_rate": _safe_div(
                    sum(1 for item in items if item.hard_violation), len(items)
                ),
                "preferred_hit@5": _safe_div(
                    sum(1 for item in items if item.preferred_hit_at_5), len(items)
                ),
            }
            for category, items in sorted(grouped.items())
        },
        "failure_counts": dict(
            Counter(
                "invalid_competitor"
                if case.invalid_competitor_articles
                else "hard_violation"
                if case.hard_violation
                else "status_failure"
                if case.expected_status != case.actual_status
                else "ok"
                for case in cases
            )
        ),
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
        "# RAG V3 Evaluation",
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
        f"- hard violation rate: `{summary['hard_violation_rate']:.4f}`",
        f"- preferred hit@1/@3/@5: `{summary['preferred_hit@1']:.4f}` / `{summary['preferred_hit@3']:.4f}` / `{summary['preferred_hit@5']:.4f}`",
        f"- preferred precision@5: `{summary['preferred_precision@5']:.4f}`",
        f"- MRR: `{summary['MRR']:.4f}`",
        f"- eligible hit@1/@5: `{summary['eligible_hit@1']:.4f}` / `{summary['eligible_hit@5']:.4f}`",
        f"- eligible coverage@5: `{summary['eligible_coverage@5']:.4f}`",
        f"- LD precision/recall/exact: `{summary['ld_mapping_precision']:.4f}` / `{summary['ld_mapping_recall']:.4f}` / `{summary['ld_mapping_exact_rate']:.4f}`",
        f"- invalid competitor rate: `{summary['invalid_competitor_rate']:.4f}`",
        f"- embedding p50/p95: `{summary['embedding_p50_ms']:.1f}` / `{summary['embedding_p95_ms']:.1f}` ms",
        f"- qdrant p50/p95: `{summary['qdrant_p50_ms']:.1f}` / `{summary['qdrant_p95_ms']:.1f}` ms",
        f"- ranking p50/p95: `{summary['ranking_p50_ms']:.1f}` / `{summary['ranking_p95_ms']:.1f}` ms",
        "",
        "## By Category",
        "",
        "| Category | Cases | Status Acc | Hard Viol. | Pref hit@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, metrics in sorted(payload["by_category"].items()):
        lines.append(
            f"| {category} | {metrics['cases']} | {metrics['status_accuracy']:.4f} | {metrics['hard_violation_rate']:.4f} | {metrics['preferred_hit@5']:.4f} |"
        )

    lines.extend(["", "## Failure Stages", ""])
    for stage, count in sorted(payload["failure_counts"].items()):
        lines.append(f"- {stage}: {count}")

    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAG-only V3.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RAG_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/rag_v3_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_rag_v3(dataset_path=args.dataset, max_cases=args.max_cases, limit=args.limit)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    print(str(args.output_json))
    return 0


__all__ = [
    "RagCaseResult",
    "evaluate_rag_v3",
    "render_report",
    "write_results_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
