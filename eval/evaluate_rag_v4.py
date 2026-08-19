"""Evaluate the V4 search pipeline with gold extraction inputs."""

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
from eval.v4_constants import CATEGORY_ORDER, DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_RAG_RESULTS_PATH
from eval.v4_schema import EvalCase, ExpectedAttributes
from rag_steel.attribute_extractor import ExtractedAttributes
from rag_steel.embeddings import create_embedder
from rag_steel.normalization import normalize_article, normalize_connection, normalize_text
from rag_steel.search_engine import SearchEngine
from rag_steel.settings import get_settings


@dataclass(slots=True)
class _DummyEmbedder:
    model_name: str = "v4-dummy"
    dimension: int = 1536
    embedding_revision: str = ""

    def embed_query(self, text: str) -> list[float]:
        del text
        return [0.0] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


class GoldV4Extractor:
    def __init__(self, expected: ExpectedAttributes) -> None:
        self.expected = expected

    def extract(self, query: str) -> ExtractedAttributes:
        del query
        return ExtractedAttributes.model_validate(
            {
                "raw_brand": self.expected.raw_brand,
                "article": self.expected.article,
                "dn": self.expected.dn,
                "pn_bar": self.expected.pn_bar,
                "connection": self.expected.connection,
                "body_material": self.expected.body_material,
                "medium": self.expected.medium,
                "control": self.expected.control,
                "temperature": self.expected.temperature,
                "length_mm": self.expected.length_mm,
                "series": self.expected.series,
            }
        )


@dataclass(slots=True)
class RagCaseResult:
    id: str
    query: str
    category: str
    expected_status: str
    actual_status: str
    resolution_mode: str | None
    expected: dict[str, Any]
    requested: dict[str, Any]
    returned_competitor_articles: list[str]
    returned_ld_articles: dict[str, list[str]]
    hard_violation: bool
    eligible_hit_at_1: bool
    eligible_hit_at_5: bool
    preferred_hit_at_1: bool
    preferred_hit_at_5: bool
    ld_mapping_exact_rate: bool
    eligible_competitor_articles: list[str]
    preferred_competitor_articles: list[str]
    returned_top5: list[dict[str, Any]]
    timing_ms: dict[str, float]
    comparison: dict[str, list[str]]
    eligible_best_rank: int | None = None
    preferred_best_rank: int | None = None

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


def _article_comparison_key(value: Any) -> str:
    normalized = normalize_article(value)
    return normalized.article_compact or normalized.article_norm or str(value).strip().casefold()


def _normalize_article_id(value: Any) -> str:
    return _article_comparison_key(value)


def _normalize_raw_article(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ")
    text = " ".join(text.split()).strip()
    return text or None


def _ld_mapping_key(value: Any) -> str:
    normalized = normalize_article(value)
    return normalized.article_norm or normalized.article_compact or str(value).strip().casefold()


def _extract_returned_articles(response: Any) -> list[str]:
    articles: list[str] = []
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if article:
            normalized = _article_comparison_key(article)
            if normalized not in articles:
                articles.append(normalized)
    return articles


def _extract_result_score(result: Any) -> float | None:
    score = getattr(result, "score", None)
    return float(score) if score is not None else None


def _extract_score_lookup(response: Any) -> dict[str, float | None]:
    lookup: dict[str, float | None] = {}
    for result in getattr(response, "results", []) or []:
        product = getattr(result, "product", None) or {}
        article = product.get("article") or product.get("article_norm")
        if not article:
            continue
        lookup[_article_comparison_key(article)] = _extract_result_score(result)
    return lookup


def _extract_returned_details(
    response: Any,
    *,
    score_lookup: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, result in enumerate(getattr(response, "results", []) or [], start=1):
        competitor = getattr(result, "competitor", None)
        if competitor is None:
            continue
        article = getattr(competitor, "article", None)
        normalized = _article_comparison_key(article) if article else None
        details.append(
            {
                "rank": getattr(result, "rank", index),
                "article": normalized or article,
                "brand": getattr(competitor, "brand", None),
                "dn": getattr(competitor, "dn", None),
                "pn_bar": getattr(competitor, "pn_bar", None),
                "connection": getattr(competitor, "connection", None),
                "score": (
                    score_lookup.get(normalized)
                    if score_lookup is not None and normalized is not None
                    else None
                ),
                "ld_articles": [
                    _article_comparison_key(item)
                    for item in getattr(result, "ld_articles", []) or []
                ],
            }
        )
    return details


def _extract_returned_ld_articles(response: Any) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        article = getattr(competitor, "article", None) if competitor is not None else None
        if not article:
            continue
        normalized_article = _ld_mapping_key(article)
        ld_articles = []
        for ld_article in getattr(result, "ld_articles", []) or []:
            normalized_ld = _ld_mapping_key(ld_article)
            if normalized_ld not in ld_articles:
                ld_articles.append(normalized_ld)
        mapping[normalized_article] = ld_articles
    return mapping


def _normalize_article_list(values: list[Any] | tuple[Any, ...] | set[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _ld_mapping_key(value)
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _best_rank(returned: list[str], target: list[str]) -> int | None:
    target_keys = {_article_comparison_key(value) for value in target}
    for rank, article in enumerate(returned, start=1):
        if article in target_keys:
            return rank
    return None


def _normalized_ld_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        _ld_mapping_key(article): _normalize_article_list(ld_articles)
        for article, ld_articles in mapping.items()
    }


def _ld_mapping_exact(
    expected_mapping: dict[str, list[str]],
    returned_mapping: dict[str, list[str]],
) -> bool:
    normalized_expected = _normalized_ld_mapping(expected_mapping)
    normalized_returned = _normalized_ld_mapping(returned_mapping)
    for article, returned_ld in normalized_returned.items():
        expected_ld = normalized_expected.get(article)
        if expected_ld is None:
            return False
        if set(returned_ld) != set(expected_ld):
            return False
    return True


def _percent_hit(returned: list[str], target: list[str], k: int) -> bool:
    returned_keys = [_article_comparison_key(value) for value in returned[:k]]
    target_keys = {_article_comparison_key(value) for value in target}
    return bool(set(returned_keys) & target_keys)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _hard_violation(expected: ExpectedAttributes, response: Any) -> bool:
    for result in getattr(response, "results", []) or []:
        competitor = getattr(result, "competitor", None)
        if competitor is None:
            continue
        competitor_brand = getattr(competitor, "brand", None)
        if expected.resolved_brand is not None and competitor_brand is not None:
            if str(expected.resolved_brand).casefold() != str(competitor_brand).casefold():
                return True
        for field in ("dn", "pn_bar", "connection"):
            expected_value = getattr(expected, field)
            actual_value = getattr(competitor, field, None)
            if expected_value is None:
                continue
            if field == "dn":
                if actual_value is None or float(expected_value) != float(actual_value):
                    return True
            elif field == "pn_bar":
                try:
                    if actual_value is None or float(actual_value) < float(expected_value):
                        return True
                except (TypeError, ValueError):
                    return True
            elif (
                actual_value is None
                or str(expected_value).casefold() != str(actual_value).casefold()
            ):
                return True
    return False


def _requested_from_response(response: Any) -> ExpectedAttributes:
    raw_requested = response.requested or {}
    if hasattr(raw_requested, "model_dump"):
        requested_dump = dict(raw_requested.model_dump())
    elif isinstance(raw_requested, dict):
        requested_dump = dict(raw_requested)
    else:
        requested_dump = dict(raw_requested)
    brand = requested_dump.get("raw_brand", requested_dump.get("brand"))
    requested = ExpectedAttributes.model_validate(
        {
            "raw_brand": brand,
            "resolved_brand": requested_dump.get("resolved_brand", brand),
            "article": requested_dump.get("article"),
            "resolved_article": requested_dump.get(
                "resolved_article", requested_dump.get("article")
            ),
            "dn": requested_dump.get("dn"),
            "pn_bar": requested_dump.get("pn_bar"),
            "connection": requested_dump.get("connection"),
            "body_material": requested_dump.get("body_material"),
            "medium": requested_dump.get("medium"),
            "control": requested_dump.get("control"),
            "temperature": requested_dump.get("temperature"),
            "length_mm": requested_dump.get("length_mm"),
            "series": requested_dump.get("series"),
        }
    )
    return ExpectedAttributes(
        raw_brand=requested.raw_brand,
        resolved_brand=requested.resolved_brand,
        article=requested.article,
        resolved_article=requested.resolved_article,
        dn=requested.dn,
        pn_bar=requested.pn_bar,
        connection=requested.connection,
        body_material=requested.body_material,
        medium=requested.medium,
        control=requested.control,
        temperature=requested.temperature,
        length_mm=requested.length_mm,
        series=requested.series,
    )


def _expected_requested(case: EvalCase) -> ExpectedAttributes:
    expected = case.expected_attributes
    return ExpectedAttributes(
        raw_brand=expected.resolved_brand,
        resolved_brand=expected.resolved_brand,
        article=expected.article,
        resolved_article=expected.resolved_article,
        dn=expected.dn,
        pn_bar=expected.pn_bar,
        connection=expected.connection,
        body_material=expected.body_material,
        medium=expected.medium,
        control=expected.control,
        temperature=expected.temperature,
        length_mm=expected.length_mm,
        series=expected.series,
    )


def _compare_requested(
    expected: ExpectedAttributes,
    actual: ExpectedAttributes,
) -> dict[str, list[str]]:
    wrong_fields: list[str] = []
    hallucinated_fields: list[str] = []
    missing_fields: list[str] = []
    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()
    for field in (
        "raw_brand",
        "resolved_brand",
        "article",
        "resolved_article",
        "dn",
        "pn_bar",
        "connection",
    ):
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
        elif field in {"raw_brand", "resolved_brand"}:
            if normalize_text(expected_value) != normalize_text(actual_value):
                wrong_fields.append(field)
        elif field == "article":
            if _normalize_raw_article(expected_value) != _normalize_raw_article(actual_value):
                wrong_fields.append(field)
        elif field == "resolved_article":
            if _normalize_article_id(expected_value) != _normalize_article_id(actual_value):
                wrong_fields.append(field)
        elif field == "connection":
            if normalize_connection(expected_value) != normalize_connection(actual_value):
                wrong_fields.append(field)
        elif str(expected_value).casefold() != str(actual_value).casefold():
            wrong_fields.append(field)
    return {
        "wrong_fields": wrong_fields,
        "hallucinated_fields": hallucinated_fields,
        "missing_fields": missing_fields,
    }


def _build_category_summary(cases: list[RagCaseResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[RagCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

    ordered_categories = [
        category for category in CATEGORY_ORDER if category in grouped
    ] + sorted(category for category in grouped if category not in CATEGORY_ORDER)

    summary: dict[str, dict[str, float | int]] = {}
    for category in ordered_categories:
        items = grouped[category]
        positive_cases = [case for case in items if case.expected_status == "exact_match"]
        summary[category] = {
            "cases": len(items),
            "positive_cases": len(positive_cases),
            "status_accuracy": _safe_div(
                sum(1 for case in items if case.expected_status == case.actual_status),
                len(items),
            ),
            "hard_violation_rate": _safe_div(
                sum(1 for case in items if case.hard_violation),
                len(items),
            ),
            "eligible_hit@1": _safe_div(
                sum(1 for case in positive_cases if case.eligible_hit_at_1),
                len(positive_cases),
            ),
            "eligible_hit@5": _safe_div(
                sum(1 for case in positive_cases if case.eligible_hit_at_5),
                len(positive_cases),
            ),
            "preferred_hit@1": _safe_div(
                sum(1 for case in positive_cases if case.preferred_hit_at_1),
                len(positive_cases),
            ),
            "preferred_hit@5": _safe_div(
                sum(1 for case in positive_cases if case.preferred_hit_at_5),
                len(positive_cases),
            ),
            "overall_pass_rate": _safe_div(
                sum(1 for case in items if _overall_pass(case)),
                len(items),
            ),
        }
    return summary


def _eligible_hit_5_failures(cases: list[RagCaseResult]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for case in cases:
        if case.expected_status != "exact_match" or case.eligible_hit_at_5:
            continue
        failures.append(
            {
                "id": case.id,
                "category": case.category,
                "query": case.query,
                "expected": {
                    "raw_brand": case.expected.get("raw_brand"),
                    "resolved_brand": case.expected.get("resolved_brand"),
                    "article": case.expected.get("article"),
                    "resolved_article": case.expected.get("resolved_article"),
                    "dn": case.expected.get("dn"),
                    "pn_bar": case.expected.get("pn_bar"),
                    "connection": case.expected.get("connection"),
                },
                "eligible_articles": case.eligible_competitor_articles,
                "preferred_articles": case.preferred_competitor_articles,
                "returned_top5": case.returned_top5,
                "resolution_mode": case.resolution_mode,
            }
        )
    return failures


def _evaluate_case(
    engine: SearchEngine,
    case: EvalCase,
) -> RagCaseResult:
    engine.attribute_extractor = GoldV4Extractor(case.expected_attributes)
    response = engine.search_v2(case.query, limit=5)
    returned_competitor_articles = _extract_returned_articles(response)
    returned_ld_articles = _extract_returned_ld_articles(response)
    returned_top5 = _extract_returned_details(response)
    actual_requested = _requested_from_response(response)
    expected_requested = _expected_requested(case)
    comparison = _compare_requested(expected_requested, actual_requested)
    hard_violation = _hard_violation(case.expected_attributes, response)

    eligible_target = case.eligible_competitor_articles
    preferred_target = case.preferred_competitor_articles
    eligible_best_rank = _best_rank(returned_competitor_articles, eligible_target)
    preferred_best_rank = _best_rank(returned_competitor_articles, preferred_target)
    score_lookup: dict[str, float | None] | None = None
    if not _percent_hit(returned_competitor_articles, eligible_target, 5):
        try:
            raw_response = engine.search(case.query, limit=5)
        except Exception:
            raw_response = None
        if raw_response is not None:
            score_lookup = _extract_score_lookup(raw_response)
            returned_top5 = _extract_returned_details(response, score_lookup=score_lookup)
    ld_mapping_exact = _ld_mapping_exact(
        case.expected_ld_articles_by_competitor,
        returned_ld_articles,
    )

    return RagCaseResult(
        id=case.id,
        query=case.query,
        category=case.category,
        expected_status=case.expected_status,
        actual_status=getattr(response, "status", "unknown"),
        resolution_mode=getattr(response, "resolution_mode", None),
        expected=case.expected_attributes.model_dump(mode="json"),
        requested=actual_requested.model_dump(mode="json"),
        returned_competitor_articles=returned_competitor_articles,
        returned_ld_articles=returned_ld_articles,
        hard_violation=hard_violation,
        eligible_hit_at_1=_percent_hit(returned_competitor_articles, eligible_target, 1),
        eligible_hit_at_5=_percent_hit(returned_competitor_articles, eligible_target, 5),
        preferred_hit_at_1=_percent_hit(returned_competitor_articles, preferred_target, 1),
        preferred_hit_at_5=_percent_hit(returned_competitor_articles, preferred_target, 5),
        ld_mapping_exact_rate=ld_mapping_exact,
        eligible_competitor_articles=eligible_target,
        preferred_competitor_articles=preferred_target,
        returned_top5=(
            returned_top5
            if not _percent_hit(returned_competitor_articles, eligible_target, 5)
            else []
        ),
        timing_ms=dict(getattr(response, "timing_ms", {}) or {}),
        comparison=comparison,
        eligible_best_rank=eligible_best_rank,
        preferred_best_rank=preferred_best_rank,
    )


def _status_accuracy(cases: list[RagCaseResult]) -> float:
    if not cases:
        return 0.0
    return _safe_div(
        sum(1 for case in cases if case.expected_status == case.actual_status),
        len(cases),
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
    if case.expected_status == "exact_match" and not case.preferred_hit_at_5:
        return False
    return True


def evaluate_rag_v4(
    *,
    dataset_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    max_cases: int | None = None,
    limit: int = 5,
    use_dummy_embedder: bool = False,
    embedder_factory: Callable[[Any], Any] | None = None,
    engine_factory: Callable[..., SearchEngine] = SearchEngine,
) -> dict[str, Any]:
    if not dataset_path.exists():
        build_v4_dataset(output_path=dataset_path)

    dataset = _load_dataset(dataset_path)
    if max_cases is not None:
        dataset = dataset[:max_cases]

    if use_dummy_embedder:
        embedder = _DummyEmbedder()
    else:
        factory = embedder_factory or create_embedder
        embedder = factory(get_settings())

    engine = engine_factory(
        embedder=embedder,
        attribute_extractor=GoldV4Extractor(ExpectedAttributes()),
    )

    cases = [_evaluate_case(engine, case) for case in dataset]
    positive_cases = [case for case in cases if case.expected_status == "exact_match"]
    summary = {
        "cases": len(cases),
        "status_accuracy": _status_accuracy(cases),
        "hard_violation_rate": _safe_div(
            sum(1 for case in cases if case.hard_violation),
            len(cases),
        ),
        "preferred_hit@1": _safe_div(
            sum(1 for case in positive_cases if case.preferred_hit_at_1),
            len(positive_cases),
        ),
        "preferred_hit@5": _safe_div(
            sum(1 for case in positive_cases if case.preferred_hit_at_5),
            len(positive_cases),
        ),
        "eligible_hit@1": _safe_div(
            sum(1 for case in positive_cases if case.eligible_hit_at_1),
            len(positive_cases),
        ),
        "eligible_hit@5": _safe_div(
            sum(1 for case in positive_cases if case.eligible_hit_at_5),
            len(positive_cases),
        ),
        "ld_mapping_exact_rate": _safe_div(
            sum(1 for case in positive_cases if case.ld_mapping_exact_rate),
            len(positive_cases),
        ),
        "overall_pass_rate": _safe_div(
            sum(1 for case in cases if _overall_pass(case)),
            len(cases),
        ),
        "strict_overall_pass_rate": _safe_div(
            sum(1 for case in cases if _strict_overall_pass(case)),
            len(cases),
        ),
    }
    by_category = _build_category_summary(cases)
    eligible_hit_5_failures = _eligible_hit_5_failures(cases)

    grouped: dict[str, list[RagCaseResult]] = defaultdict(list)
    for case in cases:
        grouped[case.resolution_mode or "unknown"].append(case)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "limit": limit,
        "summary": summary,
        "by_category": by_category,
        "by_resolution_mode": {mode: len(items) for mode, items in sorted(grouped.items())},
        "eligible_hit_5_failures": eligible_hit_5_failures,
        "failure_counts": dict(
            Counter(
                "hard_violation"
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
        "# RAG V4 Evaluation",
        "",
        "## Overall",
        f"- cases: `{summary['cases']}`",
        f"- status accuracy: `{summary['status_accuracy']:.4f}`",
        f"- hard violation rate: `{summary['hard_violation_rate']:.4f}`",
        (
            f"- preferred hit@1/@5: "
            f"`{summary['preferred_hit@1']:.4f}` / `{summary['preferred_hit@5']:.4f}`"
        ),
        (
            f"- eligible hit@1/@5: "
            f"`{summary['eligible_hit@1']:.4f}` / `{summary['eligible_hit@5']:.4f}`"
        ),
        f"- ld mapping exact rate: `{summary['ld_mapping_exact_rate']:.4f}`",
        f"- overall pass rate: `{summary['overall_pass_rate']:.4f}`",
        f"- strict overall pass rate: `{summary['strict_overall_pass_rate']:.4f}`",
        "",
        "## By Category",
    ]
    for category, metrics in payload["by_category"].items():
        lines.extend(
            [
                f"- {category}:",
                (
                    f"  - cases: `{metrics['cases']}`; "
                    f"positive_cases: `{metrics['positive_cases']}`; "
                    f"status_accuracy: `{metrics['status_accuracy']:.4f}`; "
                    f"hard_violation_rate: `{metrics['hard_violation_rate']:.4f}`"
                ),
                (
                    f"  - eligible_hit@1/@5: `{metrics['eligible_hit@1']:.4f}` / "
                    f"`{metrics['eligible_hit@5']:.4f}`"
                ),
                (
                    f"  - preferred_hit@1/@5: `{metrics['preferred_hit@1']:.4f}` / "
                    f"`{metrics['preferred_hit@5']:.4f}`"
                ),
                f"  - overall_pass_rate: `{metrics['overall_pass_rate']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Eligible Hit@5 Failures",
        ]
    )
    for case in payload.get("eligible_hit_5_failures", []):
        lines.extend(
            [
                f"- `{case['id']}` [{case['category']}] {case['query']}",
                f"  - resolution_mode: `{case['resolution_mode']}`",
                f"  - expected: `{json.dumps(case['expected'], ensure_ascii=False)}`",
                f"  - eligible: `{', '.join(case['eligible_articles'])}`",
                f"  - preferred: `{', '.join(case['preferred_articles'])}`",
                f"  - returned_top5: `{json.dumps(case['returned_top5'], ensure_ascii=False)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## By Resolution Mode",
        ]
    )
    for mode, count in sorted(payload["by_resolution_mode"].items()):
        lines.append(f"- {mode}: `{count}`")
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAG for V4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--smoke-embedder",
        action="store_true",
        help="Use a deterministic dummy embedder instead of the production embedder.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RAG_RESULTS_PATH)
    parser.add_argument("--output-md", type=Path, default=Path("eval/rag_v4_report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = evaluate_rag_v4(
        dataset_path=args.dataset,
        max_cases=args.max_cases,
        use_dummy_embedder=args.smoke_embedder,
    )
    write_results_json(payload, args.output_json)
    report = render_report(payload, args.output_md)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
