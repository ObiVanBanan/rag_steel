"""Evaluate DeepSeek extraction for V4."""

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

from eval.build_v4_eval_dataset import build_v4_dataset
from eval.v4_constants import DEFAULT_DEEPSEEK_RESULTS_PATH, DEFAULT_GOLDEN_DATASET_PATH
from eval.v4_schema import EvalCase, ExpectedAttributes
from rag_steel.attribute_extractor import create_attribute_extractor
from rag_steel.normalization import normalize_article, normalize_connection, normalize_text
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
)
from rag_steel.settings import get_settings

_EXTRACTION_FIELDS = (
    "raw_brand",
    "article",
    "dn",
    "pn_bar",
    "connection",
    "body_material",
    "medium",
    "control",
    "temperature",
    "length_mm",
    "series",
)


@dataclass(slots=True)
class DeepSeekCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    expected_resolution_mode: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    comparison: dict[str, list[str]]
    invalid_response: bool
    error_type: str | None
    error_message: str | None
    latency_ms: float

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


def _field_equal(field: str, expected: Any, actual: Any) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if field in {"dn", "pn_bar", "length_mm"}:
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    if field in {"raw_brand", "resolved_brand"}:
        return normalize_text(expected) == normalize_text(actual)
    if field in {"article", "resolved_article"}:
        expected_article = normalize_article(expected)
        actual_article = normalize_article(actual)
        return (expected_article.article_compact or expected_article.article_norm) == (
            actual_article.article_compact or actual_article.article_norm
        )
    if field == "connection":
        return normalize_connection(expected) == normalize_connection(actual)
    return normalize_text(expected) == normalize_text(actual)


def _compare_expected_actual(
    expected: ExpectedAttributes,
    actual: ExpectedAttributes,
) -> dict[str, list[str]]:
    wrong_fields: list[str] = []
    hallucinated_fields: list[str] = []
    missing_fields: list[str] = []

    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()
    for field in _EXTRACTION_FIELDS:
        expected_value = expected_dump.get(field)
        actual_value = actual_dump.get(field)
        if expected_value is None and actual_value is None:
            continue
        if expected_value is None and actual_value is not None:
            hallucinated_fields.append(field)
            continue
        if expected_value is not None and actual_value is None:
            missing_fields.append(field)
            continue
        if not _field_equal(field, expected_value, actual_value):
            wrong_fields.append(field)

    return {
        "wrong_fields": wrong_fields,
        "hallucinated_fields": hallucinated_fields,
        "missing_fields": missing_fields,
    }


def _actual_from_extracted(extracted: Any) -> ExpectedAttributes:
    if hasattr(extracted, "model_dump"):
        payload = extracted.model_dump()
    elif isinstance(extracted, dict):
        payload = dict(extracted)
    else:
        payload = dict(extracted)
    return ExpectedAttributes.model_validate(payload)


def _evaluate_case(
    case: EvalCase,
    *,
    extractor_factory: Callable[[Any], Any],
    settings: Any,
) -> DeepSeekCaseResult:
    started = perf_counter()
    invalid_response = False
    error_type: str | None = None
    error_message: str | None = None

    extractor = extractor_factory(settings)
    try:
        extracted = extractor.extract(case.query)
        actual = _actual_from_extracted(extracted)
    except DeepSeekInvalidResponseError as exc:
        invalid_response = True
        error_type = "invalid_response"
        error_message = str(exc)
        actual = ExpectedAttributes()
    except DeepSeekTimeoutError as exc:
        error_type = "timeout"
        error_message = str(exc)
        actual = ExpectedAttributes()
    except DeepSeekUpstreamError as exc:
        error_type = "upstream_error"
        error_message = str(exc)
        actual = ExpectedAttributes()
    except DeepSeekConfigurationError as exc:
        error_type = "configuration_error"
        error_message = str(exc)
        actual = ExpectedAttributes()
    except Exception as exc:
        error_type = "unexpected_error"
        error_message = f"{type(exc).__name__}: {exc}"
        actual = ExpectedAttributes()

    latency_ms = (perf_counter() - started) * 1000.0
    comparison = _compare_expected_actual(case.expected_attributes, actual)
    return DeepSeekCaseResult(
        id=case.id,
        query=case.query,
        category=case.category,
        expected_status=case.expected_status,
        expected_resolution_mode=case.expected_resolution_mode,
        expected=case.expected_attributes.model_dump(mode="json"),
        actual=actual.model_dump(mode="json"),
        comparison=comparison,
        invalid_response=invalid_response,
        error_type=error_type,
        error_message=error_message,
        latency_ms=latency_ms,
    )


def _field_accuracy(cases: list[DeepSeekCaseResult], field: str) -> float:
    expected_cases = [case for case in cases if case.expected.get(field) is not None]
    if not expected_cases:
        return 0.0
    correct = sum(
        1
        for case in expected_cases
        if _field_equal(field, case.expected.get(field), case.actual.get(field))
    )
    return correct / len(expected_cases)


def _hallucination_rate(cases: list[DeepSeekCaseResult], fields: tuple[str, ...]) -> float:
    if not cases:
        return 0.0
    hallucinated = 0
    for case in cases:
        if any(
            case.expected.get(field) is None and case.actual.get(field) is not None
            for field in fields
        ):
            hallucinated += 1
    return hallucinated / len(cases)


def evaluate_deepseek_v4(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    extractor_factory: Callable[[Any], Any] = create_attribute_extractor,
    settings: Any | None = None,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v4_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    settings = settings or get_settings()
    cases = [
        _evaluate_case(case, extractor_factory=extractor_factory, settings=settings)
        for case in dataset
    ]

    summary = {
        "cases": len(cases),
        "raw_brand_accuracy": _field_accuracy(cases, "raw_brand"),
        "article_accuracy": _field_accuracy(cases, "article"),
        "dn_accuracy": _field_accuracy(cases, "dn"),
        "pn_accuracy": _field_accuracy(cases, "pn_bar"),
        "connection_accuracy": _field_accuracy(cases, "connection"),
        "hard_hallucination_rate": _hallucination_rate(
            cases,
            ("raw_brand", "dn", "pn_bar", "connection"),
        ),
        "soft_hallucination_rate": _hallucination_rate(
            cases,
            ("body_material", "medium", "control", "temperature", "length_mm", "series"),
        ),
        "brand_hallucination_rate": _hallucination_rate(cases, ("raw_brand",)),
        "article_hallucination_rate": _hallucination_rate(cases, ("article",)),
        "latency_p50_ms": sorted(case.latency_ms for case in cases)[len(cases) // 2]
        if cases
        else 0.0,
        "latency_p95_ms": sorted(case.latency_ms for case in cases)[
            max(0, int(len(cases) * 0.95) - 1)
        ]
        if cases
        else 0.0,
    }

    failures = [
        {
            "id": case.id,
            "query": case.query,
            "expected": case.expected,
            "actual": case.actual,
            "wrong_fields": case.comparison["wrong_fields"],
            "hallucinated_fields": case.comparison["hallucinated_fields"],
            "missing_fields": case.comparison["missing_fields"],
            "error_type": case.error_type,
            "error_message": case.error_message,
            "latency_ms": round(case.latency_ms, 3),
        }
        for case in cases
        if case.comparison["wrong_fields"]
        or case.comparison["hallucinated_fields"]
        or case.comparison["missing_fields"]
        or case.error_type is not None
    ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    grouped: dict[str, list[DeepSeekCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "deepseek_model": settings.deepseek_model,
        "summary": summary,
        "cases": [case.to_dict() for case in cases],
        "failures": failures,
        "failure_counts": dict(
            Counter(
                case.error_type
                or ("attribute_mismatch" if case.comparison["wrong_fields"] else "ok")
                for case in cases
            )
        ),
        "by_category": {
            category: {
                "cases": len(items),
                "raw_brand_accuracy": _field_accuracy(items, "raw_brand"),
                "article_accuracy": _field_accuracy(items, "article"),
            }
            for category, items in sorted(grouped.items())
        },
    }
    return payload


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def render_report(payload: dict[str, Any], output_path: Path) -> str:
    summary = payload["summary"]
    lines = [
        "# DeepSeek V4 Evaluation",
        "",
        "## Overall",
        f"- cases: `{summary['cases']}`",
        f"- raw brand accuracy: `{summary['raw_brand_accuracy']:.4f}`",
        f"- article accuracy: `{summary['article_accuracy']:.4f}`",
        (
            f"- dn / pn / connection accuracy: "
            f"`{summary['dn_accuracy']:.4f}` / "
            f"`{summary['pn_accuracy']:.4f}` / "
            f"`{summary['connection_accuracy']:.4f}`"
        ),
        f"- brand hallucination rate: `{summary['brand_hallucination_rate']:.4f}`",
        f"- article hallucination rate: `{summary['article_hallucination_rate']:.4f}`",
        f"- hard hallucination rate: `{summary['hard_hallucination_rate']:.4f}`",
        f"- soft hallucination rate: `{summary['soft_hallucination_rate']:.4f}`",
        "",
        "## Failures",
    ]
    for failure in payload["failures"][:20]:
        failure_id = failure["id"]
        query = failure["query"]
        wrong_fields = failure["wrong_fields"]
        missing_fields = failure["missing_fields"]
        hallucinated_fields = failure["hallucinated_fields"]
        error_type = failure["error_type"] or "-"
        lines.append(
            f"- `{failure_id}` `{query}` wrong={wrong_fields} "
            f"missing={missing_fields} hallucinated={hallucinated_fields} "
            f"error={error_type}"
        )
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek extraction for V4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DEEPSEEK_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/deepseek_v4_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_deepseek_v4(dataset_path=args.dataset, max_cases=args.max_cases)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
