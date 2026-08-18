"""Evaluate the deterministic V4 resolution layer."""

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

from eval.build_v4_eval_dataset import build_v4_dataset
from eval.v4_constants import DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_RESOLUTION_RESULTS_PATH
from eval.v4_schema import EvalCase, ExpectedAttributes
from rag_steel.search_engine import SearchEngine
from rag_steel.settings import get_settings


@dataclass(slots=True)
class _DummyEmbedder:
    model_name: str = "v4-dummy"
    dimension: int = 3
    embedding_revision: str = ""

    def embed_query(self, text: str) -> list[float]:
        del text
        return [0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


@dataclass(slots=True)
class _DummyExtractor:
    def extract(self, query: str) -> Any:
        del query
        return {}


@dataclass(slots=True)
class ResolutionCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    expected_resolution_mode: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    actual_status: str
    actual_resolution_mode: str | None
    reason_code: str | None
    comparison: dict[str, list[str]]
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


def _status_from_reason(reason_code: str | None) -> str:
    if reason_code in {"COMPETITOR_BRAND_REQUIRED", "UNSUPPORTED_COMPETITOR_BRAND"}:
        return "cannot_process"
    if reason_code in {"ARTICLE_NOT_FOUND", "ARTICLE_AMBIGUOUS", "IDENTITY_CONFLICT"}:
        return "not_found"
    return "exact_match"


def _compare_expected_actual(
    expected: ExpectedAttributes,
    actual: ExpectedAttributes,
) -> dict[str, list[str]]:
    wrong_fields: list[str] = []
    hallucinated_fields: list[str] = []
    missing_fields: list[str] = []

    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()
    for field in ("raw_brand", "resolved_brand", "article", "resolved_article"):
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
        if str(expected_value).casefold() != str(actual_value).casefold():
            wrong_fields.append(field)
    for field in ("dn", "pn_bar", "connection"):
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
        if field in {"dn", "pn_bar"}:
            try:
                if float(expected_value) != float(actual_value):
                    wrong_fields.append(field)
            except (TypeError, ValueError):
                wrong_fields.append(field)
        elif str(expected_value).casefold() != str(actual_value).casefold():
            wrong_fields.append(field)

    return {
        "wrong_fields": wrong_fields,
        "hallucinated_fields": hallucinated_fields,
        "missing_fields": missing_fields,
    }


def _evaluate_case(
    engine: SearchEngine,
    case: EvalCase,
) -> ResolutionCaseResult:
    started = datetime.now(timezone.utc)
    resolution = engine.query_resolver.resolve(
        raw_brand=case.expected_attributes.raw_brand,
        raw_article=case.expected_attributes.article,
        dn=case.expected_attributes.dn,
        pn_bar=case.expected_attributes.pn_bar,
        connection=case.expected_attributes.connection,
    )
    latency_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0

    resolved_brand = resolution.brand.canonical if resolution.brand else None
    resolved_article = resolution.article.article if resolution.article else None
    actual = ExpectedAttributes(
        raw_brand=resolution.brand.raw,
        resolved_brand=resolved_brand,
        article=resolution.article.raw if resolution.article else None,
        resolved_article=resolved_article,
        dn=case.expected_attributes.dn,
        pn_bar=case.expected_attributes.pn_bar,
        connection=case.expected_attributes.connection,
        body_material=case.expected_attributes.body_material,
        medium=case.expected_attributes.medium,
        control=case.expected_attributes.control,
        temperature=case.expected_attributes.temperature,
        length_mm=case.expected_attributes.length_mm,
        series=case.expected_attributes.series,
    )
    comparison = _compare_expected_actual(case.expected_attributes, actual)
    actual_status = _status_from_reason(resolution.reason_code)
    return ResolutionCaseResult(
        id=case.id,
        query=case.query,
        category=case.category,
        expected_status=case.expected_status,
        expected_resolution_mode=case.expected_resolution_mode,
        expected=case.expected_attributes.model_dump(mode="json"),
        actual=actual.model_dump(mode="json"),
        actual_status=actual_status,
        actual_resolution_mode=resolution.resolution_mode,
        reason_code=resolution.reason_code,
        comparison=comparison,
        latency_ms=latency_ms,
    )


def _overall_resolution_accuracy(cases: list[ResolutionCaseResult]) -> float:
    if not cases:
        return 0.0
    correct = 0
    for case in cases:
        if case.expected_status != case.actual_status:
            continue
        if case.expected_resolution_mode != case.actual_resolution_mode:
            continue
        if case.comparison["wrong_fields"] or case.comparison["missing_fields"]:
            continue
        correct += 1
    return correct / len(cases)


def _false_correction_rate(cases: list[ResolutionCaseResult]) -> float:
    if not cases:
        return 0.0
    false_corrections = 0
    for case in cases:
        expected = case.expected
        actual = case.actual
        if expected.get("resolved_brand") is None and actual.get("resolved_brand") is not None:
            false_corrections += 1
            continue
        if expected.get("resolved_article") is None and actual.get("resolved_article") is not None:
            false_corrections += 1
    return false_corrections / len(cases)


def evaluate_resolution_v4(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    engine_factory: Callable[..., SearchEngine] = SearchEngine,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v4_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    settings = get_settings()
    engine = engine_factory(
        embedder=_DummyEmbedder(dimension=settings.embedding_dimension),
        attribute_extractor=_DummyExtractor(),
    )

    cases = [_evaluate_case(engine, case) for case in dataset]
    summary = {
        "cases": len(cases),
        "brand_exact_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "brand_exact"
                and case.actual_resolution_mode == "brand_exact"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "brand_exact"
            ),
        ),
        "brand_fuzzy_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "brand_fuzzy"
                and case.actual_resolution_mode == "brand_fuzzy"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "brand_fuzzy"
            ),
        ),
        "article_exact_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "article_exact"
                and case.actual_resolution_mode == "article_exact"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "article_exact"
            ),
        ),
        "article_fuzzy_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "article_fuzzy"
                and case.actual_resolution_mode == "article_fuzzy"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "article_fuzzy"
            ),
        ),
        "ambiguity_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "article_ambiguous"
                and case.actual_resolution_mode == "article_ambiguous"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "article_ambiguous"
            ),
        ),
        "identity_conflict_accuracy": _safe_div(
            sum(
                1
                for case in cases
                if case.expected_resolution_mode == "identity_conflict"
                and case.actual_resolution_mode == "identity_conflict"
            ),
            sum(
                1 for case in cases if case.expected_resolution_mode == "identity_conflict"
            ),
        ),
        "overall_resolution_accuracy": _overall_resolution_accuracy(cases),
        "false_correction_rate": _false_correction_rate(cases),
    }

    grouped: dict[str, list[ResolutionCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.expected_resolution_mode].append(case)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "summary": summary,
        "cases": [case.to_dict() for case in cases],
        "by_resolution_mode": {mode: len(items) for mode, items in sorted(grouped.items())},
        "failure_counts": dict(
            Counter(
                case.reason_code or ("ok" if not case.comparison["wrong_fields"] else "comparison")
                for case in cases
            )
        ),
    }
    return payload


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_results_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def render_report(payload: dict[str, Any], output_path: Path) -> str:
    summary = payload["summary"]
    lines = [
        "# Resolution V4 Evaluation",
        "",
        "## Overall",
        f"- cases: `{summary['cases']}`",
        f"- brand exact accuracy: `{summary['brand_exact_accuracy']:.4f}`",
        f"- brand fuzzy accuracy: `{summary['brand_fuzzy_accuracy']:.4f}`",
        f"- article exact accuracy: `{summary['article_exact_accuracy']:.4f}`",
        f"- article fuzzy accuracy: `{summary['article_fuzzy_accuracy']:.4f}`",
        f"- ambiguity accuracy: `{summary['ambiguity_accuracy']:.4f}`",
        f"- identity conflict accuracy: `{summary['identity_conflict_accuracy']:.4f}`",
        (
            f"- overall resolution accuracy: "
            f"`{summary['overall_resolution_accuracy']:.4f}`"
        ),
        f"- false correction rate: `{summary['false_correction_rate']:.4f}`",
        "",
        "## By Mode",
    ]
    for mode, count in sorted(payload["by_resolution_mode"].items()):
        lines.append(f"- {mode}: `{count}`")
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate resolution for V4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESOLUTION_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/resolution_v4_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_resolution_v4(dataset_path=args.dataset, max_cases=args.max_cases)
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
