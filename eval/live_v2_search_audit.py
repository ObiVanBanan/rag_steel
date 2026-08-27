"""Run a live regression audit against the V2 search API."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from rag_steel.normalization import (
    normalize_article,
    normalize_body_material,
    normalize_brand,
    normalize_connection,
    normalize_control,
)

DEFAULT_DATASET_PATH = Path("eval/data/live_v2_search_audit.jsonl")
DEFAULT_RESULTS_PATH = Path("eval/results/live_v2_search_audit_latest.json")
DEFAULT_REPORT_PATH = Path("eval/live_v2_search_audit_report.md")

CLASSIFICATIONS = (
    "PASS",
    "PRODUCT_BUG",
    "DATA_MISSING",
    "PRESENTATION_ISSUE",
    "EVAL_CHECKER_BUG",
    "EXPECTED_NORMALIZATION",
)


@dataclass(slots=True)
class AuditRecord:
    id: str
    category: str
    query: str
    suite: str = "full"
    limit: int = 5
    expected_status: str | None = None
    expected_statuses: list[str] = field(default_factory=list)
    expected_reason_code: str | None = None
    expected_article: str | None = None
    expected_article_identity_any_of: list[str] = field(default_factory=list)
    forbid_article: str | None = None
    expected_brand: str | None = None
    expected_dn: float | None = None
    expected_requested_dn: float | None = None
    expected_pn_min: float | None = None
    expected_connection: str | None = None
    expected_requested_connection: str | None = None
    expected_body_material_contains: str | None = None
    forbid_body_material_contains: str | None = None
    expected_control: str | None = None
    expected_not_article: bool = False
    expected_search_mode: str | None = None
    expected_failure_code: str | None = None
    expected_classification: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class CaseResult:
    record: AuditRecord
    http_status: int | None
    total_ms: float
    response: dict[str, Any]
    issues: list[str]
    classification: str
    metrics: dict[str, bool | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "suite": self.record.suite,
            "category": self.record.category,
            "query": self.record.query,
            "expected": {
                key: value
                for key, value in asdict(self.record).items()
                if key.startswith("expected_") or key.startswith("forbid_")
            },
            "http_status": self.http_status,
            "total_ms": self.total_ms,
            "classification": self.classification,
            "issues": self.issues,
            "metrics": self.metrics,
            "response": self.response,
        }


def load_dataset(path: Path, *, suite: str = "full") -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        record = AuditRecord(**payload)
        if suite == "smoke" and record.suite != "smoke":
            continue
        records.append(record)
    return records


def article_compact(value: Any) -> str | None:
    normalized = normalize_article(value)
    return normalized.article_compact


def article_identity_equal(left: Any, right: Any) -> bool:
    left_compact = article_compact(left)
    right_compact = article_compact(right)
    return left_compact is not None and left_compact == right_compact


def pn_meets_minimum(candidate_pn: Any, requested_pn: Any) -> bool:
    try:
        return float(candidate_pn) >= float(requested_pn)
    except (TypeError, ValueError):
        return False


def has_duplicate_ld_articles(result: dict[str, Any]) -> bool:
    articles = [
        article_compact(article) or str(article) for article in result.get("ld_articles") or []
    ]
    return len(articles) != len(set(articles))


def expected_status_matches(record: AuditRecord, actual_status: str | None) -> bool:
    if record.expected_statuses:
        return actual_status in set(record.expected_statuses)
    if record.expected_status is not None:
        return actual_status == record.expected_status
    return True


def expected_reason_matches(record: AuditRecord, response: dict[str, Any]) -> bool:
    if record.expected_reason_code is None:
        return True
    reason = response.get("reason") or {}
    return reason.get("code") == record.expected_reason_code


def metric_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split()).casefold()


def _first_competitor(response: dict[str, Any]) -> dict[str, Any] | None:
    results = response.get("results") or []
    if not results:
        return None
    competitor = results[0].get("competitor")
    return competitor if isinstance(competitor, dict) else None


def _timing(response: dict[str, Any], key: str) -> float | None:
    value = (response.get("timing_ms") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _all_competitors(response: dict[str, Any]) -> list[dict[str, Any]]:
    competitors: list[dict[str, Any]] = []
    for result in response.get("results") or []:
        competitor = result.get("competitor")
        if isinstance(competitor, dict):
            competitors.append(competitor)
    return competitors


def check_response(
    record: AuditRecord, http_status: int | None, response: dict[str, Any]
) -> CaseResult:
    issues: list[str] = []
    metrics: dict[str, bool | None] = {
        "article_identity": None,
        "dn_extraction": None,
        "hard_constraints": None,
        "conflict_detection": None,
        "no_duplicate_results": None,
    }
    status = response.get("status")
    requested = response.get("requested") or {}
    top = _first_competitor(response)
    results = response.get("results") or []

    if http_status != 200:
        issues.append(f"HTTP expected 200 got {http_status}")
    if not expected_status_matches(record, status):
        expected = record.expected_statuses or record.expected_status
        issues.append(f"status expected {expected} got {status}")
    if not expected_reason_matches(record, response):
        reason = response.get("reason") or {}
        issues.append(f"reason expected {record.expected_reason_code} got {reason.get('code')}")

    if record.expected_requested_dn is not None:
        metrics["dn_extraction"] = requested.get("dn") == float(record.expected_requested_dn)
        if not metrics["dn_extraction"]:
            issues.append(
                f"requested.dn expected {record.expected_requested_dn} "
                f"got {requested.get('dn')}"
            )

    if record.expected_not_article:
        requested_article = requested.get("article")
        if requested_article is not None:
            issues.append(
                "query should not enter article path, "
                f"got requested.article={requested_article}"
            )

    article_expectations = [value for value in [record.expected_article] if value]
    article_expectations.extend(record.expected_article_identity_any_of)
    if article_expectations and status == "exact_match":
        actual_article = (top or {}).get("article") or requested.get("resolved_article")
        metrics["article_identity"] = any(
            article_identity_equal(actual_article, expected) for expected in article_expectations
        )
        if not metrics["article_identity"]:
            issues.append(f"article identity expected {article_expectations} got {actual_article}")
        elif record.expected_article and actual_article != record.expected_article:
            issues.append(
                f"raw article alias differs: expected {record.expected_article} "
                f"got {actual_article}"
            )

    if record.forbid_article and status == "exact_match":
        returned_articles = [
            competitor.get("article")
            for competitor in _all_competitors(response)
            if competitor.get("article") is not None
        ]
        if any(
            article_identity_equal(article, record.forbid_article) for article in returned_articles
        ):
            issues.append(f"forbidden article returned: {record.forbid_article}")

    duplicate_free = True
    if len(results) > min(record.limit, 20):
        duplicate_free = False
        issues.append(f"result count {len(results)} exceeds limit/min20")
    for index, result in enumerate(results, start=1):
        if has_duplicate_ld_articles(result):
            duplicate_free = False
            issues.append(f"duplicate ld_articles in result #{index}")
    metrics["no_duplicate_results"] = duplicate_free if results else None

    hard_checks: list[bool] = []
    for competitor in _all_competitors(response):
        if record.expected_brand is not None:
            ok = normalize_brand(competitor.get("brand")) == normalize_brand(record.expected_brand)
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    f"brand expected {record.expected_brand} got {competitor.get('brand')}"
                )
        if record.expected_dn is not None:
            ok = competitor.get("dn") == float(record.expected_dn)
            hard_checks.append(ok)
            if not ok:
                issues.append(f"dn expected {record.expected_dn} got {competitor.get('dn')}")
        if record.expected_pn_min is not None:
            ok = pn_meets_minimum(competitor.get("pn_bar"), record.expected_pn_min)
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    f"pn expected >= {record.expected_pn_min} got {competitor.get('pn_bar')}"
                )
        if record.expected_connection is not None:
            ok = normalize_connection(competitor.get("connection")) == normalize_connection(
                record.expected_connection
            )
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    f"connection expected {record.expected_connection} "
                    f"got {competitor.get('connection')}"
                )
        if record.expected_body_material_contains is not None:
            expected = normalize_body_material(record.expected_body_material_contains)
            actual = normalize_body_material(competitor.get("body_material"))
            ok = expected is not None and actual is not None and expected in actual
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    "body_material expected contains "
                    f"{record.expected_body_material_contains} "
                    f"got {competitor.get('body_material')}"
                )
        if record.forbid_body_material_contains is not None:
            forbidden = normalize_body_material(record.forbid_body_material_contains)
            actual = normalize_body_material(competitor.get("body_material"))
            ok = forbidden is None or actual is None or forbidden not in actual
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    "body_material forbidden contains "
                    f"{record.forbid_body_material_contains} got {competitor.get('body_material')}"
                )
        if record.expected_control is not None and status == "exact_match":
            ok = normalize_control(competitor.get("control")) == normalize_control(
                record.expected_control
            )
            hard_checks.append(ok)
            if not ok:
                issues.append(
                    f"control expected {record.expected_control} got {competitor.get('control')}"
                )

    if hard_checks:
        metrics["hard_constraints"] = all(hard_checks)

    if record.expected_requested_connection is not None:
        ok = normalize_connection(requested.get("connection")) == normalize_connection(
            record.expected_requested_connection
        )
        if not ok:
            issues.append(
                "requested.connection expected "
                f"{record.expected_requested_connection} got {requested.get('connection')}"
            )

    if record.expected_failure_code and record.expected_classification == "PRODUCT_BUG":
        metrics["conflict_detection"] = status in {"cannot_process", "not_found"}
        if not metrics["conflict_detection"]:
            issues.append(record.expected_failure_code)

    classification = classify_case(record, response, issues)
    return CaseResult(
        record=record,
        http_status=http_status,
        total_ms=float(response.get("_client_total_ms", 0.0)),
        response={key: value for key, value in response.items() if key != "_client_total_ms"},
        issues=issues,
        classification=classification,
        metrics=metrics,
    )


def classify_case(record: AuditRecord, response: dict[str, Any], issues: list[str]) -> str:
    expected = record.expected_classification
    raw_alias_only = issues and all(
        issue.startswith("raw article alias differs") for issue in issues
    )
    if expected == "DATA_MISSING":
        return "DATA_MISSING"
    if expected == "EXPECTED_NORMALIZATION":
        return "EXPECTED_NORMALIZATION" if not issues else "PRODUCT_BUG"
    if expected == "EVAL_CHECKER_BUG":
        return "EVAL_CHECKER_BUG" if raw_alias_only or not issues else "PRODUCT_BUG"
    if raw_alias_only:
        return "EVAL_CHECKER_BUG"
    if expected == "PRODUCT_BUG" and issues:
        return "PRODUCT_BUG"
    if expected == "PRODUCT_BUG" and response.get("status") in {"cannot_process", "not_found"}:
        return "PASS"
    if issues:
        return expected if expected in CLASSIFICATIONS and expected != "PASS" else "PRODUCT_BUG"
    return "PASS"


def call_search(
    base_url: str, record: AuditRecord, *, timeout: float
) -> tuple[int | None, dict[str, Any]]:
    url = base_url.rstrip("/") + "/v2/search"
    body = json.dumps({"query": record.query, "limit": record.limit}, ensure_ascii=False).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_client_total_ms"] = (time.perf_counter() - started) * 1000.0
            return response.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        payload["_client_total_ms"] = (time.perf_counter() - started) * 1000.0
        return exc.code, payload
    except Exception as exc:  # pragma: no cover - exercised only by live failures
        return None, {
            "error": repr(exc),
            "_client_total_ms": (time.perf_counter() - started) * 1000.0,
        }


def call_json(base_url: str, path: str, *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
            return {
                "http_status": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except Exception as exc:  # pragma: no cover - exercised only by live failures
        return {"error": repr(exc)}


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((percentile_value / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def latency_summary(cases: list[CaseResult], *, mode: str) -> dict[str, float | None]:
    values: list[float] = []
    for case in cases:
        is_article = case.response.get("requested", {}).get(
            "article"
        ) is not None or case.record.category in {
            "article_identity",
            "article_conflict",
            "short_numeric_article",
        }
        if mode == "article" and not is_article:
            continue
        if mode == "semantic" and is_article:
            continue
        values.append(case.total_ms)
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def ratio_metric(cases: list[CaseResult], key: str) -> float | None:
    values = [case.metrics[key] for case in cases if case.metrics.get(key) is not None]
    if not values:
        return None
    return sum(1 for value in values if value is True) / len(values)


def summarize(cases: list[CaseResult]) -> dict[str, Any]:
    classifications = Counter(case.classification for case in cases)
    by_category: dict[str, dict[str, Any]] = {}
    for category, grouped_cases_iter in _group_by_category(cases).items():
        grouped_cases = list(grouped_cases_iter)
        by_category[category] = {
            "total": len(grouped_cases),
            "classifications": dict(Counter(case.classification for case in grouped_cases)),
            "article_identity_accuracy": ratio_metric(grouped_cases, "article_identity"),
            "extraction_dn_accuracy": ratio_metric(grouped_cases, "dn_extraction"),
            "hard_constraint_accuracy": ratio_metric(grouped_cases, "hard_constraints"),
            "conflict_detection_accuracy": ratio_metric(grouped_cases, "conflict_detection"),
            "no_duplicate_result_rate": ratio_metric(grouped_cases, "no_duplicate_results"),
        }
    return {
        "total": len(cases),
        "passed": classifications["PASS"],
        "failed": len(cases) - classifications["PASS"],
        "product_bugs": classifications["PRODUCT_BUG"],
        "data_missing": classifications["DATA_MISSING"],
        "presentation_issues": classifications["PRESENTATION_ISSUE"],
        "checker_issues": classifications["EVAL_CHECKER_BUG"],
        "expected_normalization": classifications["EXPECTED_NORMALIZATION"],
        "classifications": dict(classifications),
        "article_identity_accuracy": ratio_metric(cases, "article_identity"),
        "extraction_dn_accuracy": ratio_metric(cases, "dn_extraction"),
        "hard_constraint_accuracy": ratio_metric(cases, "hard_constraints"),
        "conflict_detection_accuracy": ratio_metric(cases, "conflict_detection"),
        "no_duplicate_result_rate": ratio_metric(cases, "no_duplicate_results"),
        "latency": {
            "article": latency_summary(cases, mode="article"),
            "semantic": latency_summary(cases, mode="semantic"),
        },
        "by_category": by_category,
    }


def _group_by_category(cases: list[CaseResult]) -> dict[str, list[CaseResult]]:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.record.category].append(case)
    return grouped


def render_metric(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Live V2 Search Audit",
        "",
        "## Scope",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- base_url: `{payload['base_url']}`",
        "- endpoint: `POST /v2/search`",
        f"- suite: `{payload['suite']}`",
        f"- dataset: `{payload['dataset_path']}`",
        "",
        "## Health",
        "",
        f"- live: `{payload['health'].get('/health/live')}`",
        f"- ready: `{payload['health'].get('/health/ready')}`",
        "",
        "## Summary",
        "",
        f"- total: `{summary['total']}`",
        f"- passed: `{summary['passed']}`",
        f"- failed: `{summary['failed']}`",
        f"- product_bugs: `{summary['product_bugs']}`",
        f"- data_missing: `{summary['data_missing']}`",
        f"- presentation_issues: `{summary['presentation_issues']}`",
        f"- checker_issues: `{summary['checker_issues']}`",
        f"- expected_normalization: `{summary['expected_normalization']}`",
        "",
        "## Metrics",
        "",
        f"- extraction_dn_accuracy: `{render_metric(summary['extraction_dn_accuracy'])}`",
        f"- article_identity_accuracy: `{render_metric(summary['article_identity_accuracy'])}`",
        f"- hard_constraint_accuracy: `{render_metric(summary['hard_constraint_accuracy'])}`",
        f"- conflict_detection_accuracy: `{render_metric(summary['conflict_detection_accuracy'])}`",
        f"- no_duplicate_result_rate: `{render_metric(summary['no_duplicate_result_rate'])}`",
        "",
        "## Latency",
        "",
        "| Mode | p50 ms | p95 ms | max ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode, stats in summary["latency"].items():
        lines.append(
            f"| {mode} | {render_metric(stats['p50'])} | "
            f"{render_metric(stats['p95'])} | {render_metric(stats['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Breakdown",
            "",
            "| Category | Total | Classifications | DN Acc | Article Acc | "
            "Hard Acc | Conflict Acc | Dedup Rate |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, stats in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {stats['total']} | `{stats['classifications']}` | "
            f"{render_metric(stats['extraction_dn_accuracy'])} | "
            f"{render_metric(stats['article_identity_accuracy'])} | "
            f"{render_metric(stats['hard_constraint_accuracy'])} | "
            f"{render_metric(stats['conflict_detection_accuracy'])} | "
            f"{render_metric(stats['no_duplicate_result_rate'])} |"
        )

    lines.extend(["", "## Findings", ""])
    non_pass = [case for case in payload["cases"] if case["classification"] != "PASS"]
    if not non_pass:
        lines.append("No non-PASS classifications.")
    for case in non_pass:
        response = case["response"]
        requested = response.get("requested")
        reason = response.get("reason")
        lines.extend(
            [
                f"### {case['classification']} - {case['id']}",
                "",
                f"- category: `{case['category']}`",
                f"- query: `{case['query']}`",
                f"- status: `{response.get('status')}`",
                f"- reason: `{reason}`",
                f"- requested: `{requested}`",
                f"- issues: `{case['issues']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Gaps",
            "",
            "- This audit exercises the live API only; unit tests for the search "
            "architecture are separate.",
            "- `expected_search_mode` is stored in the dataset, but the API response "
            "does not expose search mode yet, so the runner does not hard-fail on that field.",
            "- Metrics with no applicable denominator are rendered as `unavailable`, not `0.0000`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    base_url: str,
    dataset_path: Path,
    suite: str,
    timeout: float,
) -> dict[str, Any]:
    records = load_dataset(dataset_path, suite=suite)
    health = {
        "/health/live": call_json(base_url, "/health/live"),
        "/health/ready": call_json(base_url, "/health/ready"),
    }
    cases: list[CaseResult] = []
    for record in records:
        http_status, response = call_search(base_url, record, timeout=timeout)
        cases.append(check_response(record, http_status, response))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "dataset_path": str(dataset_path),
        "suite": suite,
        "health": health,
        "summary": summarize(cases),
        "cases": [case.to_dict() for case in cases],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live /v2/search regression audit.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--suite", choices=("full", "smoke"), default="full")
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = run_audit(
        base_url=args.base_url,
        dataset_path=args.dataset,
        suite=args.suite,
        timeout=args.timeout,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(render_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
