"""Evaluate SearchEngine.search_v2() against a deterministic V2 dataset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient

from rag_steel.normalization import normalize_article
from rag_steel.query_constraints import QueryConstraints, extract_query_constraints
from rag_steel.search_engine import SearchEngine
from rag_steel.settings import QDRANT_COLLECTION_ALIAS, QDRANT_URL, SOURCE_CANDIDATE_LIMIT

DEFAULT_DATASET_PATH = Path("eval/v2_queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORT_PATH = Path("eval/v2_report.md")


@dataclass(slots=True)
class EvalRecord:
    id: str
    query: str
    category: str
    gold_mode: str
    expected_status: str
    expected_constraints: dict[str, Any]
    eligible_competitor_articles: list[str]
    expected_ld_articles_by_competitor: dict[str, list[str]]


@dataclass(slots=True)
class EvaluatedCase:
    id: str
    query: str
    category: str
    gold_mode: str
    expected_constraints: dict[str, Any]
    actual_constraints: dict[str, Any]
    expected_status: str
    actual_status: str
    eligible_competitor_articles: list[str]
    raw_candidate_count: int
    raw_eligible_count: int
    filtered_candidate_count: int
    filtered_eligible_count: int
    returned_competitor_articles: list[str]
    invalid_returned_articles: list[str]
    ld_mismatches: list[dict[str, Any]]
    timing_ms: dict[str, float]
    failure_stage: str
    raw_candidate_articles_top20: list[str]
    filtered_articles_top20: list[str]
    parser_constraint_exact: bool


def _load_dataset(path: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            EvalRecord(
                id=str(payload["id"]),
                query=str(payload["query"]),
                category=str(payload["category"]),
                gold_mode=str(payload["gold_mode"]),
                expected_status=str(payload["expected_status"]),
                expected_constraints=dict(payload["expected_constraints"]),
                eligible_competitor_articles=[
                    str(item) for item in payload["eligible_competitor_articles"]
                ],
                expected_ld_articles_by_competitor={
                    str(key): [str(article) for article in value]
                    for key, value in dict(payload["expected_ld_articles_by_competitor"]).items()
                },
            )
        )
    return records


def _normalize_article_id(value: Any) -> str:
    normalized = normalize_article(value)
    return normalized.article_norm or str(value).strip().casefold()


def _constraints_from_model(constraints: QueryConstraints) -> dict[str, Any]:
    return constraints.model_dump()


def _constraints_equal(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    keys = set(expected) | set(actual)
    return all(expected.get(key) == actual.get(key) for key in keys)


def _parser_relevant(case: EvaluatedCase) -> bool:
    return case.gold_mode in {"constraints", "parser_only"}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _match_constraints(record: EvalRecord) -> dict[str, Any]:
    return dict(record.expected_constraints)


def _returned_competitor_article(result: Any) -> str | None:
    competitor = getattr(result, "competitor", None)
    if competitor is None:
        return None
    article = getattr(competitor, "article", None)
    if not article:
        return None
    return _normalize_article_id(article)


def _returned_ld_articles(result: Any) -> list[str]:
    return [_normalize_article_id(article) for article in getattr(result, "ld_articles", [])]


def _classify_failure_stage(case: EvaluatedCase) -> str:
    if case.gold_mode == "parser_only":
        return "ok" if case.parser_constraint_exact else "parser_failure"
    if case.expected_status == "not_found":
        return "false_exact_match" if case.actual_status == "exact_match" else "ok"
    if _parser_relevant(case) and not case.parser_constraint_exact:
        return "parser_failure"
    if case.raw_eligible_count == 0:
        return "retrieval_failure"
    if case.filtered_eligible_count == 0:
        return "strict_filter_failure"
    returned_set = set(case.returned_competitor_articles)
    eligible_set = set(case.eligible_competitor_articles)
    if not (returned_set & eligible_set):
        return "response_failure"
    if case.ld_mismatches:
        return "ld_mapping_failure"
    return "ok"


def _hit_at_k(cases: list[EvaluatedCase], k: int) -> float:
    positive = [
        case
        for case in cases
        if case.expected_status == "exact_match" and case.gold_mode != "parser_only"
    ]
    hits = 0
    for case in positive:
        if set(case.returned_competitor_articles[:k]) & set(case.eligible_competitor_articles):
            hits += 1
    return _safe_div(hits, len(positive))


def _precision_at_k(cases: list[EvaluatedCase], k: int) -> float:
    positive = [
        case
        for case in cases
        if case.expected_status == "exact_match" and case.gold_mode != "parser_only"
    ]
    numer = 0
    denom = 0
    for case in positive:
        eligible = set(case.eligible_competitor_articles)
        returned = case.returned_competitor_articles[:k]
        numer += sum(1 for article in returned if article in eligible)
        denom += len(returned)
    return _safe_div(numer, denom)


def _coverage_at_k(cases: list[EvaluatedCase], k: int) -> float:
    positive = [
        case
        for case in cases
        if case.expected_status == "exact_match" and case.gold_mode != "parser_only"
    ]
    numer = 0.0
    denom = len(positive)
    for case in positive:
        eligible = set(case.eligible_competitor_articles)
        if not eligible:
            continue
        denominator = min(k, len(eligible))
        numer += _safe_div(
            len(set(case.returned_competitor_articles[:k]) & eligible),
            denominator,
        )
    return _safe_div(numer, denom)


def _invalid_competitor_rate(cases: list[EvaluatedCase]) -> float:
    numer = sum(
        len(case.invalid_returned_articles) for case in cases if case.gold_mode != "parser_only"
    )
    denom = sum(
        len(case.returned_competitor_articles) for case in cases if case.gold_mode != "parser_only"
    )
    return _safe_div(numer, denom)


def _false_exact_match_rate(cases: list[EvaluatedCase]) -> float:
    negatives = [case for case in cases if case.expected_status == "not_found"]
    numer = sum(1 for case in negatives if case.actual_status == "exact_match")
    return _safe_div(numer, len(negatives))


def _not_found_precision(cases: list[EvaluatedCase]) -> float:
    predicted = [
        case
        for case in cases
        if case.actual_status == "not_found" and case.gold_mode != "parser_only"
    ]
    correct = sum(1 for case in predicted if case.expected_status == "not_found")
    return _safe_div(correct, len(predicted))


def _not_found_recall(cases: list[EvaluatedCase]) -> float:
    expected = [case for case in cases if case.expected_status == "not_found"]
    correct = sum(1 for case in expected if case.actual_status == "not_found")
    return _safe_div(correct, len(expected))


def _ld_mapping_micro(
    cases: list[EvaluatedCase],
    dataset: dict[str, EvalRecord],
) -> tuple[float, float]:
    returned_total = 0
    expected_total = 0
    hit_total = 0
    for case in cases:
        if case.gold_mode == "parser_only":
            continue
        record = dataset[case.id]
        for article in case.returned_competitor_articles:
            if article not in record.eligible_competitor_articles:
                continue
            expected = set(record.expected_ld_articles_by_competitor.get(article, []))
            mismatch = next(
                (item for item in case.ld_mismatches if item["competitor_article"] == article),
                None,
            )
            if mismatch is not None:
                returned = set(mismatch["returned"])
            else:
                returned = set(expected)
            hit_total += len(returned & expected)
            returned_total += len(returned)
            expected_total += len(expected)
    return _safe_div(hit_total, returned_total), _safe_div(hit_total, expected_total)


def _status_accuracy(cases: list[EvaluatedCase]) -> float:
    relevant = [case for case in cases if case.gold_mode != "parser_only"]
    correct = sum(1 for case in relevant if case.expected_status == case.actual_status)
    return _safe_div(correct, len(relevant))


def _constraint_exact_match_rate(cases: list[EvaluatedCase]) -> float:
    relevant = [case for case in cases if _parser_relevant(case)]
    return _safe_div(sum(1 for case in relevant if case.parser_constraint_exact), len(relevant))


def _retrieval_any_hit_rate(cases: list[EvaluatedCase]) -> float:
    positive = [
        case
        for case in cases
        if case.expected_status == "exact_match" and case.gold_mode != "parser_only"
    ]
    hits = sum(1 for case in positive if case.raw_eligible_count > 0)
    return _safe_div(hits, len(positive))


def _source_candidate_recall(cases: list[EvaluatedCase]) -> float:
    positive = [
        case
        for case in cases
        if case.expected_status == "exact_match" and case.gold_mode != "parser_only"
    ]
    numer = 0.0
    for case in positive:
        eligible_count = len(case.eligible_competitor_articles)
        if eligible_count == 0:
            continue
        numer += _safe_div(case.raw_eligible_count, eligible_count)
    return _safe_div(numer, len(positive))


def _per_category(cases: list[EvaluatedCase]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[EvaluatedCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    payload: dict[str, dict[str, float | int]] = {}
    for category, items in sorted(grouped.items()):
        payload[category] = {
            "cases": len(items),
            "parser_exact_match_rate": _constraint_exact_match_rate(items),
            "status_accuracy": _status_accuracy(items),
            "competitor_hit@5": _hit_at_k(items, 5),
            "invalid_competitor_rate": _invalid_competitor_rate(items),
        }
    return payload


def _latency_summary(cases: list[EvaluatedCase]) -> dict[str, float]:
    embedding = [case.timing_ms.get("embedding", 0.0) for case in cases if case.timing_ms]
    qdrant = [case.timing_ms.get("qdrant", 0.0) for case in cases if case.timing_ms]
    ranking = [case.timing_ms.get("ranking", 0.0) for case in cases if case.timing_ms]
    total = [case.timing_ms.get("total", 0.0) for case in cases if case.timing_ms]
    return {
        "embedding_p50_ms": _percentile(embedding, 50),
        "embedding_p95_ms": _percentile(embedding, 95),
        "qdrant_p50_ms": _percentile(qdrant, 50),
        "qdrant_p95_ms": _percentile(qdrant, 95),
        "ranking_p50_ms": _percentile(ranking, 50),
        "ranking_p95_ms": _percentile(ranking, 95),
        "total_p50_ms": _percentile(total, 50),
        "total_p95_ms": _percentile(total, 95),
    }


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


def _capture_search_v2_case(engine: SearchEngine, record: EvalRecord, limit: int) -> EvaluatedCase:
    constraints = extract_query_constraints(record.query)
    actual_constraints = _constraints_from_model(constraints)
    parser_exact = _constraints_equal(_match_constraints(record), actual_constraints)

    if record.gold_mode == "parser_only":
        case = EvaluatedCase(
            id=record.id,
            query=record.query,
            category=record.category,
            gold_mode=record.gold_mode,
            expected_constraints=record.expected_constraints,
            actual_constraints=actual_constraints,
            expected_status=record.expected_status,
            actual_status="parser_only",
            eligible_competitor_articles=record.eligible_competitor_articles,
            raw_candidate_count=0,
            raw_eligible_count=0,
            filtered_candidate_count=0,
            filtered_eligible_count=0,
            returned_competitor_articles=[],
            invalid_returned_articles=[],
            ld_mismatches=[],
            timing_ms={},
            failure_stage="ok",
            raw_candidate_articles_top20=[],
            filtered_articles_top20=[],
            parser_constraint_exact=parser_exact,
        )
        case.failure_stage = _classify_failure_stage(case)
        return case

    captured: dict[str, Any] = {}
    engine_class = type(engine)
    original_query_points = engine_class._query_points

    def wrapped_query_points(
        self: SearchEngine,
        client: QdrantClient,
        query: str,
        dense_vector: list[float],
        **kwargs: Any,
    ) -> Any:
        response = original_query_points(self, client, query, dense_vector, **kwargs)
        captured["points"] = list(self._extract_points(response))
        return response

    engine_class._query_points = wrapped_query_points  # type: ignore[assignment]
    try:
        response = engine.search_v2(record.query, limit=limit)
    finally:
        engine_class._query_points = original_query_points  # type: ignore[assignment]

    raw_points = list(captured.get("points", []))
    filtered_points = list(engine._filter_points_by_constraints(raw_points, constraints))
    raw_articles = [
        _normalize_article_id(
            engine._build_source_product(engine._extract_payload(point)).get("article")
        )
        for point in raw_points
        if engine._build_source_product(engine._extract_payload(point)).get("article")
    ]
    filtered_articles = [
        _normalize_article_id(
            engine._build_source_product(engine._extract_payload(point)).get("article")
        )
        for point in filtered_points
        if engine._build_source_product(engine._extract_payload(point)).get("article")
    ]
    eligible_set = set(record.eligible_competitor_articles)
    raw_eligible_articles = set(raw_articles) & eligible_set
    filtered_eligible_articles = set(filtered_articles) & eligible_set
    returned_articles = [
        article
        for article in (_returned_competitor_article(result) for result in response.results)
        if article
    ]
    invalid = [article for article in returned_articles if article not in eligible_set]
    ld_mismatches: list[dict[str, Any]] = []
    for result in response.results:
        article = _returned_competitor_article(result)
        if not article or article not in eligible_set:
            continue
        expected = record.expected_ld_articles_by_competitor.get(article, [])
        returned = _returned_ld_articles(result)
        if set(expected) != set(returned):
            ld_mismatches.append(
                {
                    "competitor_article": article,
                    "expected": expected,
                    "returned": returned,
                    "hits": len(set(expected) & set(returned)),
                    "returned_count": len(returned),
                }
            )
    case = EvaluatedCase(
        id=record.id,
        query=record.query,
        category=record.category,
        gold_mode=record.gold_mode,
        expected_constraints=record.expected_constraints,
        actual_constraints=actual_constraints,
        expected_status=record.expected_status,
        actual_status=response.status,
        eligible_competitor_articles=record.eligible_competitor_articles,
        raw_candidate_count=len(raw_points),
        raw_eligible_count=len(raw_eligible_articles),
        filtered_candidate_count=len(filtered_points),
        filtered_eligible_count=len(filtered_eligible_articles),
        returned_competitor_articles=returned_articles,
        invalid_returned_articles=invalid,
        ld_mismatches=ld_mismatches,
        timing_ms=dict(response.timing_ms),
        failure_stage="ok",
        raw_candidate_articles_top20=raw_articles[:20],
        filtered_articles_top20=filtered_articles[:20],
        parser_constraint_exact=parser_exact,
    )
    case.failure_stage = _classify_failure_stage(case)
    return case


def evaluate_v2(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    limit: int = 5,
    max_cases: int | None = None,
    qdrant_url: str = QDRANT_URL,
    collection_alias: str = QDRANT_COLLECTION_ALIAS,
    engine_factory: Callable[..., SearchEngine] = SearchEngine,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    engine = engine_factory(qdrant_url=qdrant_url, collection_alias=collection_alias)
    ready, readiness = engine.readiness_status()
    if not ready:
        raise RuntimeError(json.dumps(readiness, ensure_ascii=False))

    client = engine._get_client()
    collection_name = readiness["resolved_collection_name"]
    point_count_before = int(client.count(collection_name=collection_name, exact=True).count)

    cases = [_capture_search_v2_case(engine, record, limit) for record in dataset]

    point_count_after = int(client.count(collection_name=collection_name, exact=True).count)
    if point_count_before != point_count_after:
        raise RuntimeError("Qdrant point count changed during V2 evaluation")

    dataset_map = {record.id: record for record in dataset}
    ld_precision, ld_recall = _ld_mapping_micro(cases, dataset_map)
    summary = {
        "constraint_exact_match_rate": _constraint_exact_match_rate(cases),
        "status_accuracy": _status_accuracy(cases),
        "retrieval_any_hit_rate": _retrieval_any_hit_rate(cases),
        "source_candidate_recall": _source_candidate_recall(cases),
        "competitor_hit@1": _hit_at_k(cases, 1),
        "competitor_hit@5": _hit_at_k(cases, 5),
        "competitor_precision@5": _precision_at_k(cases, 5),
        "competitor_coverage@5": _coverage_at_k(cases, 5),
        "invalid_competitor_rate": _invalid_competitor_rate(cases),
        "false_exact_match_rate": _false_exact_match_rate(cases),
        "not_found_precision": _not_found_precision(cases),
        "not_found_recall": _not_found_recall(cases),
        "ld_mapping_precision": ld_precision,
        "ld_mapping_recall": ld_recall,
        **_latency_summary(cases),
    }
    failure_counts = dict(Counter(case.failure_stage for case in cases))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "collection_alias": collection_alias,
        "resolved_collection_name": collection_name,
        "point_count": point_count_before,
        "source_candidate_limit": SOURCE_CANDIDATE_LIMIT,
        "limit": limit,
        "summary": summary,
        "per_category": _per_category(cases),
        "failure_counts": failure_counts,
        "cases": [asdict(case) for case in cases],
        "readiness": readiness,
    }


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def render_report(payload: dict[str, Any], output_path: Path = DEFAULT_REPORT_PATH) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    lines = [
        "# V2 Search Evaluation",
        "",
        "## Run",
        f"- commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- collection: `{payload['resolved_collection_name']}`",
        f"- point count: `{payload['point_count']}`",
        f"- dataset: `{payload['dataset_path']}`",
        f"- query count: `{len(cases)}`",
        "",
        "## Overall",
        f"- constraint exact match: `{summary['constraint_exact_match_rate']:.4f}`",
        f"- retrieval any-hit: `{summary['retrieval_any_hit_rate']:.4f}`",
        f"- source candidate recall: `{summary['source_candidate_recall']:.4f}`",
        f"- competitor hit@1: `{summary['competitor_hit@1']:.4f}`",
        f"- competitor hit@5: `{summary['competitor_hit@5']:.4f}`",
        f"- competitor precision@5: `{summary['competitor_precision@5']:.4f}`",
        f"- competitor coverage@5: `{summary['competitor_coverage@5']:.4f}`",
        f"- invalid competitor rate: `{summary['invalid_competitor_rate']:.4f}`",
        f"- false exact match rate: `{summary['false_exact_match_rate']:.4f}`",
        (
            f"- not-found precision/recall: `{summary['not_found_precision']:.4f}` / "
            f"`{summary['not_found_recall']:.4f}`"
        ),
        (
            f"- LD precision/recall: `{summary['ld_mapping_precision']:.4f}` / "
            f"`{summary['ld_mapping_recall']:.4f}`"
        ),
        (
            f"- p50/p95 latency total: `{summary['total_p50_ms']:.1f}` / "
            f"`{summary['total_p95_ms']:.1f}` ms"
        ),
        "",
        "## By Category",
        "",
        "| Category | Cases | Parser Exact | Status Accuracy | Hit@5 | Invalid Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, metrics in payload["per_category"].items():
        lines.append(
            f"| {category} | {metrics['cases']} | {metrics['parser_exact_match_rate']:.4f} | "
            f"{metrics['status_accuracy']:.4f} | {metrics['competitor_hit@5']:.4f} | "
            f"{metrics['invalid_competitor_rate']:.4f} |"
        )

    lines.extend(["", "## Failure Stages", ""])
    for stage, count in sorted(payload["failure_counts"].items()):
        lines.append(f"- {stage}: {count}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["failure_stage"]].append(case)
    lines.extend(["", "## Worst Cases", ""])
    for stage in (
        "parser_failure",
        "retrieval_failure",
        "strict_filter_failure",
        "response_failure",
        "false_exact_match",
        "ld_mapping_failure",
    ):
        stage_cases = grouped.get(stage, [])[:20]
        if not stage_cases:
            continue
        lines.append(f"### {stage}")
        for case in stage_cases:
            lines.append(
                f"- `{case['id']}` `{case['category']}` `{case['query']}` "
                f"(status `{case['actual_status']}`, "
                f"invalid={len(case['invalid_returned_articles'])})"
            )
        lines.append("")

    if summary["false_exact_match_rate"] > 0:
        lines.append("CRITICAL: false_exact_match_rate > 0")
    if summary["invalid_competitor_rate"] > 0:
        lines.append("CRITICAL: invalid_competitor_rate > 0")
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SearchEngine.search_v2()")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Dataset JSONL path",
    )
    parser.add_argument("--limit", type=int, default=5, help="Public result limit")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional smoke-run case limit",
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="Qdrant URL")
    parser.add_argument(
        "--collection-alias",
        default=QDRANT_COLLECTION_ALIAS,
        help="Collection alias",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_v2(
        dataset_path=args.dataset,
        limit=args.limit,
        max_cases=args.max_cases,
        qdrant_url=args.qdrant_url,
        collection_alias=args.collection_alias,
    )
    results_path = DEFAULT_RESULTS_DIR / f"v2_{payload['run_id']}.json"
    write_results_json(payload, results_path)
    report = render_report(payload, DEFAULT_REPORT_PATH)
    print(report)
    print(str(results_path))
    return 0


__all__ = [
    "_classify_failure_stage",
    "_coverage_at_k",
    "_false_exact_match_rate",
    "_hit_at_k",
    "_invalid_competitor_rate",
    "_ld_mapping_micro",
    "_not_found_precision",
    "_not_found_recall",
    "_percentile",
    "_precision_at_k",
    "_source_candidate_recall",
    "evaluate_v2",
]


if __name__ == "__main__":
    raise SystemExit(main())
