"""Evaluate only brand detection and DeepSeek attribute extraction for V3."""

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

from rag_steel.attribute_extractor import create_attribute_extractor
from rag_steel.brand_gate import detect_competitor_brand
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
)
from rag_steel.settings import get_settings

from eval.build_v3_eval_dataset import build_v3_dataset
from eval.v3_common import compare_expected_actual, hard_exact_match, _percentile, _safe_div
from eval.v3_constants import DEFAULT_DEEPSEEK_RESULTS_PATH, DEFAULT_GOLDEN_DATASET_PATH
from eval.v3_schema import EvalCase, ExpectedAttributes


@dataclass(slots=True)
class DeepSeekCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    brand_expected: str | None
    brand_actual: str | None
    brand_correct: bool
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


def _actual_attributes(brand: str | None, extracted: Any) -> ExpectedAttributes:
    if hasattr(extracted, "model_dump"):
        extracted_data = extracted.model_dump()
    elif isinstance(extracted, dict):
        extracted_data = dict(extracted)
    else:
        extracted_data = ExpectedAttributes.model_validate(extracted).model_dump()
    extracted_data["brand"] = brand
    return ExpectedAttributes.model_validate(extracted_data)


def _evaluate_case(
    case: EvalCase,
    *,
    brand_detector: Callable[[str], str | None],
    extractor_factory: Callable[[Any], Any],
    settings: Any,
) -> DeepSeekCaseResult:
    started = perf_counter()
    brand_actual = brand_detector(case.query)
    invalid_response = False
    error_type: str | None = None
    error_message: str | None = None

    if brand_actual is None:
        actual = ExpectedAttributes()
    else:
        extractor = extractor_factory(settings)
        try:
            extracted = extractor.extract(case.query)
            actual = _actual_attributes(brand_actual, extracted)
        except DeepSeekInvalidResponseError as exc:
            invalid_response = True
            error_type = "invalid_response"
            error_message = str(exc)
            actual = ExpectedAttributes(brand=brand_actual)
        except DeepSeekTimeoutError as exc:
            error_type = "timeout"
            error_message = str(exc)
            actual = ExpectedAttributes(brand=brand_actual)
        except DeepSeekUpstreamError as exc:
            error_type = "upstream_error"
            error_message = str(exc)
            actual = ExpectedAttributes(brand=brand_actual)
        except DeepSeekConfigurationError as exc:
            error_type = "configuration_error"
            error_message = str(exc)
            actual = ExpectedAttributes(brand=brand_actual)
        except Exception as exc:
            error_type = "unexpected_error"
            error_message = f"{type(exc).__name__}: {exc}"
            actual = ExpectedAttributes(brand=brand_actual)

    latency_ms = (perf_counter() - started) * 1000.0
    comparison = compare_expected_actual(case.expected_attributes, actual)
    return DeepSeekCaseResult(
        id=case.id,
        query=case.query,
        category=case.category,
        expected_status=case.expected_status,
        expected=case.expected_attributes.model_dump(mode="json"),
        actual=actual.model_dump(mode="json"),
        brand_expected=case.expected_attributes.brand,
        brand_actual=brand_actual,
        brand_correct=case.expected_attributes.brand == brand_actual,
        comparison=comparison,
        invalid_response=invalid_response,
        error_type=error_type,
        error_message=error_message,
        latency_ms=latency_ms,
    )


def _extraction_cases(cases: list[DeepSeekCaseResult]) -> list[DeepSeekCaseResult]:
    return [case for case in cases if case.expected_status == "exact_match"]


def _field_accuracy(cases: list[DeepSeekCaseResult], field: str) -> float:
    expected_cases = [case for case in cases if case.expected.get(field) is not None]
    if not expected_cases:
        return 0.0
    correct = sum(1 for case in expected_cases if case.expected.get(field) == case.actual.get(field))
    return _safe_div(correct, len(expected_cases))


def _hallucination_rate(cases: list[DeepSeekCaseResult], fields: tuple[str, ...]) -> float:
    total = len(cases)
    if total == 0:
        return 0.0
    hallucinated = 0
    for case in cases:
        if any(
            case.expected.get(field) is None and case.actual.get(field) is not None
            for field in fields
        ):
            hallucinated += 1
    return _safe_div(hallucinated, total)


def _missing_rate(cases: list[DeepSeekCaseResult], fields: tuple[str, ...]) -> float:
    total = len(cases)
    if total == 0:
        return 0.0
    missing = 0
    for case in cases:
        if any(
            case.expected.get(field) is not None and case.actual.get(field) is None
            for field in fields
        ):
            missing += 1
    return _safe_div(missing, total)


def _brand_false_positive_rate(cases: list[DeepSeekCaseResult]) -> float:
    negatives = [case for case in cases if case.brand_expected is None]
    if not negatives:
        return 0.0
    false_positives = sum(1 for case in negatives if case.brand_actual is not None)
    return _safe_div(false_positives, len(negatives))


def _brand_false_negative_rate(cases: list[DeepSeekCaseResult]) -> float:
    positives = [case for case in cases if case.brand_expected is not None]
    if not positives:
        return 0.0
    false_negatives = sum(1 for case in positives if case.brand_actual is None)
    return _safe_div(false_negatives, len(positives))


def evaluate_deepseek_v3(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    brand_detector: Callable[[str], str | None] = detect_competitor_brand,
    extractor_factory: Callable[[Any], Any] = create_attribute_extractor,
    settings: Any | None = None,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v3_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    settings = settings or get_settings()
    cases = [
        _evaluate_case(
            case,
            brand_detector=brand_detector,
            extractor_factory=extractor_factory,
            settings=settings,
        )
        for case in dataset
    ]
    extraction_cases = _extraction_cases(cases)

    summary = {
        "cases": len(cases),
        "extraction_cases": len(extraction_cases),
        "brand_accuracy": _safe_div(sum(1 for case in cases if case.brand_correct), len(cases)),
        "brand_false_positive_rate": _brand_false_positive_rate(cases),
        "brand_false_negative_rate": _brand_false_negative_rate(cases),
        "hard_exact_match_rate": _safe_div(
            sum(
                1
                for case in extraction_cases
                if hard_exact_match(
                    ExpectedAttributes.model_validate(case.expected),
                    ExpectedAttributes.model_validate(case.actual),
                )
            ),
            len(extraction_cases),
        ),
        "dn_accuracy": _field_accuracy(extraction_cases, "dn"),
        "pn_accuracy": _field_accuracy(extraction_cases, "pn_bar"),
        "connection_accuracy": _field_accuracy(extraction_cases, "connection"),
        "body_material_accuracy": _field_accuracy(extraction_cases, "body_material"),
        "medium_accuracy": _field_accuracy(extraction_cases, "medium"),
        "control_accuracy": _field_accuracy(extraction_cases, "control"),
        "temperature_accuracy": _field_accuracy(extraction_cases, "temperature"),
        "length_accuracy": _field_accuracy(extraction_cases, "length_mm"),
        "series_accuracy": _field_accuracy(extraction_cases, "series"),
        "article_accuracy": _field_accuracy(extraction_cases, "article"),
        "hard_hallucination_rate": _hallucination_rate(
            extraction_cases, ("brand", "dn", "pn_bar", "connection")
        ),
        "soft_hallucination_rate": _hallucination_rate(
            extraction_cases,
            ("body_material", "medium", "control", "temperature", "length_mm", "series", "article"),
        ),
        "hard_missing_rate": _missing_rate(
            extraction_cases, ("brand", "dn", "pn_bar", "connection")
        ),
        "soft_missing_rate": _missing_rate(
            extraction_cases,
            ("body_material", "medium", "control", "temperature", "length_mm", "series", "article"),
        ),
        "invalid_response_rate": _safe_div(
            sum(1 for case in cases if case.invalid_response), len(cases)
        ),
        "timeout_rate": _safe_div(
            sum(1 for case in cases if case.error_type == "timeout"), len(cases)
        ),
        "upstream_error_rate": _safe_div(
            sum(1 for case in cases if case.error_type == "upstream_error"), len(cases)
        ),
        "configuration_error_rate": _safe_div(
            sum(1 for case in cases if case.error_type == "configuration_error"), len(cases)
        ),
        "unexpected_error_rate": _safe_div(
            sum(1 for case in cases if case.error_type == "unexpected_error"), len(cases)
        ),
        "latency_p50_ms": _percentile([case.latency_ms for case in cases], 50),
        "latency_p95_ms": _percentile([case.latency_ms for case in cases], 95),
        "latency_p99_ms": _percentile([case.latency_ms for case in cases], 99),
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
        or case.invalid_response
        or not case.brand_correct
    ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "deepseek_model": settings.deepseek_model,
        "embedding_model": getattr(settings, "embedding_model", None),
        "embedding_dimension": getattr(settings, "embedding_dimension", None),
        "qdrant_alias": getattr(settings, "qdrant_collection_alias", None),
        "resolved_collection": None,
        "limit": None,
        "source_candidate_limit": getattr(settings, "source_candidate_limit", None),
        "timestamp": run_id,
        "summary": summary,
        "cases": [case.to_dict() for case in cases],
        "failures": failures,
        "failure_counts": dict(
            Counter(
                case.error_type
                or ("brand_gate_failure" if not case.brand_correct else "attribute_mismatch")
                for case in cases
                if case.error_type
                or case.invalid_response
                or not case.brand_correct
                or case.comparison["wrong_fields"]
                or case.comparison["hallucinated_fields"]
                or case.comparison["missing_fields"]
            )
        ),
    }
    return payload


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def render_report(payload: dict[str, Any], output_path: Path) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    failures = payload["failures"]
    lines = [
        "# DeepSeek V3 Evaluation",
        "",
        "## Run",
        f"- commit: `{payload.get('git_commit') or 'unknown'}`",
        f"- dataset: `{payload['dataset_path']}`",
        f"- cases: `{len(cases)}`",
        f"- model: `{payload['deepseek_model']}`",
        "",
        "## Overall",
        f"- brand accuracy: `{summary['brand_accuracy']:.4f}`",
        f"- brand false positive rate: `{summary['brand_false_positive_rate']:.4f}`",
        f"- brand false negative rate: `{summary['brand_false_negative_rate']:.4f}`",
        f"- extraction cases: `{summary['extraction_cases']}`",
        f"- hard exact match rate: `{summary['hard_exact_match_rate']:.4f}`",
        f"- dn / pn / connection accuracy: `{summary['dn_accuracy']:.4f}` / `{summary['pn_accuracy']:.4f}` / `{summary['connection_accuracy']:.4f}`",
        f"- hard hallucination rate: `{summary['hard_hallucination_rate']:.4f}`",
        f"- soft hallucination rate: `{summary['soft_hallucination_rate']:.4f}`",
        f"- hard missing rate: `{summary['hard_missing_rate']:.4f}`",
        f"- soft missing rate: `{summary['soft_missing_rate']:.4f}`",
        f"- invalid / timeout / upstream / config / unexpected: `{summary['invalid_response_rate']:.4f}` / `{summary['timeout_rate']:.4f}` / `{summary['upstream_error_rate']:.4f}` / `{summary['configuration_error_rate']:.4f}` / `{summary['unexpected_error_rate']:.4f}`",
        f"- latency p50 / p95 / p99: `{summary['latency_p50_ms']:.1f}` / `{summary['latency_p95_ms']:.1f}` / `{summary['latency_p99_ms']:.1f}` ms",
        "",
        "## By Category",
        "",
        "| Category | Cases | Brand Acc | Hard Exact | Invalid |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["category"]].append(case)
    for category in sorted(grouped):
        items = grouped[category]
        category_cases = len(items)
        category_extraction_cases = [item for item in items if item["expected_status"] == "exact_match"]
        category_brand_acc = _safe_div(sum(1 for item in items if item["brand_correct"]), category_cases)
        category_hard_exact = _safe_div(
            sum(
                1
                for item in category_extraction_cases
                if not item["comparison"]["wrong_fields"]
                and not item["comparison"]["hallucinated_fields"]
                and not item["comparison"]["missing_fields"]
            ),
            len(category_extraction_cases),
        )
        category_invalid = _safe_div(sum(1 for item in items if item["invalid_response"]), category_cases)
        lines.append(
            f"| {category} | {category_cases} | {category_brand_acc:.4f} | {category_hard_exact:.4f} | {category_invalid:.4f} |"
        )

    lines.extend(["", "## Worst Failures", ""])
    for failure in failures[:20]:
        lines.append(
            f"- `{failure['id']}` `{failure['query']}` wrong={failure['wrong_fields']} missing={failure['missing_fields']} hallucinated={failure['hallucinated_fields']} error={failure['error_type'] or '-'}"
        )

    if summary["hard_exact_match_rate"] < 0.95:
        lines.append("")
        lines.append("FAILED: hard_exact_match_rate < 0.95")
    if summary["hard_hallucination_rate"] > 0.01:
        lines.append("")
        lines.append("FAILED: hard_hallucination_rate > 0.01")

    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek extraction for V3.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DEEPSEEK_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/deepseek_v3_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_deepseek_v3(dataset_path=args.dataset, max_cases=args.max_cases)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    print(str(args.output_json))
    return 0


__all__ = [
    "DeepSeekCaseResult",
    "evaluate_deepseek_v3",
    "render_report",
    "write_results_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
