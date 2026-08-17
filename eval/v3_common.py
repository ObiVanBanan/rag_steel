"""Shared helpers for V3 evaluation, dataset generation, and reporting."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from rag_steel.attribute_extractor import ExtractedAttributes
from rag_steel.normalization import (
    normalize_article,
    normalize_body_material,
    normalize_brand,
    normalize_connection,
    normalize_control,
    normalize_dn,
    normalize_length,
    normalize_medium,
    normalize_pn_bar,
    normalize_temperature,
    normalize_text,
)
from rag_steel.schemas import SteelProductDocument

from eval.v3_constants import HARD_FIELDS, SOFT_FIELDS
from eval.v3_schema import ExpectedAttributes

_SERIES_RE = re.compile(r"(?:series|серия|серии)\s*([0-9]{1,4})", re.IGNORECASE)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _normalize_article_id(value: Any) -> str:
    normalized = normalize_article(value)
    return normalized.article_norm or str(value).strip().casefold()


def _normalize_article_list(values: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        token = _normalize_article_id(value)
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _strip_to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = normalize_text(value)
    return text


def _field_matches(field: str, expected: Any, actual: Any) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if field in {"dn", "pn_bar", "length_mm"}:
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    if field == "brand":
        return normalize_brand(expected) == normalize_brand(actual)
    if field == "connection":
        return normalize_connection(expected) == normalize_connection(actual)
    if field == "body_material":
        return normalize_body_material(expected) == normalize_body_material(actual)
    if field == "medium":
        return normalize_medium(expected) == normalize_medium(actual)
    if field == "control":
        return normalize_control(expected) == normalize_control(actual)
    if field == "temperature":
        return normalize_temperature(expected) == normalize_temperature(actual)
    if field == "series":
        return _strip_to_text(expected) == _strip_to_text(actual)
    if field == "article":
        return _normalize_article_id(expected) == _normalize_article_id(actual)
    return _strip_to_text(expected) == _strip_to_text(actual)


def compare_expected_actual(
    expected: ExpectedAttributes,
    actual: ExpectedAttributes,
    *,
    fields: Sequence[str] = HARD_FIELDS + SOFT_FIELDS,
) -> dict[str, list[str]]:
    wrong_fields: list[str] = []
    hallucinated_fields: list[str] = []
    missing_fields: list[str] = []

    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()

    for field in fields:
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
        if not _field_matches(field, expected_value, actual_value):
            wrong_fields.append(field)

    return {
        "wrong_fields": wrong_fields,
        "hallucinated_fields": hallucinated_fields,
        "missing_fields": missing_fields,
    }


def hard_exact_match(expected: ExpectedAttributes, actual: ExpectedAttributes) -> bool:
    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()
    for field in HARD_FIELDS:
        if not _field_matches(field, expected_dump.get(field), actual_dump.get(field)):
            return False
    return True


def matches_hard_constraints(expected: ExpectedAttributes, actual: ExpectedAttributes) -> bool:
    expected_dump = expected.model_dump()
    actual_dump = actual.model_dump()
    for field in HARD_FIELDS:
        expected_value = expected_dump.get(field)
        actual_value = actual_dump.get(field)
        if expected_value is None:
            continue
        if not _field_matches(field, expected_value, actual_value):
            return False
    return True


def has_soft_constraints(expected: ExpectedAttributes) -> bool:
    expected_dump = expected.model_dump()
    return any(expected_dump.get(field) is not None for field in SOFT_FIELDS)


def find_hard_eligible_documents(
    documents: Sequence[SteelProductDocument],
    brand: str | None,
    dn: float | None,
    pn_bar: float | None,
    connection: str | None,
) -> list[SteelProductDocument]:
    eligible: list[SteelProductDocument] = []
    brand_norm = normalize_brand(brand)
    connection_norm = normalize_connection(connection) if connection is not None else None

    for document in documents:
        if brand_norm is not None and normalize_brand(document.brand) != brand_norm:
            continue
        if dn is not None and document.dn != float(dn):
            continue
        if pn_bar is not None and document.pn_bar != float(pn_bar):
            continue
        if connection_norm is not None and normalize_connection(document.connection) != connection_norm:
            continue
        eligible.append(document)
    return eligible


def _soft_match_score(
    document: SteelProductDocument,
    expected: ExpectedAttributes,
) -> int:
    score = 0
    if expected.body_material is not None and normalize_body_material(document.body_material) == normalize_body_material(expected.body_material):
        score += 1
    if expected.medium is not None and normalize_medium(document.medium) == normalize_medium(expected.medium):
        score += 1
    if expected.control is not None and normalize_control(document.control) == normalize_control(expected.control):
        score += 1
    if expected.temperature is not None and normalize_temperature(document.temperature) == normalize_temperature(expected.temperature):
        score += 1
    if expected.length_mm is not None and document.length_mm == float(expected.length_mm):
        score += 1
    if expected.article is not None:
        expected_article = _normalize_article_id(expected.article)
        haystack = " ".join(
            value
            for value in (document.article, document.article_norm, document.article_compact, document.name)
            if value
        )
        if expected_article in {
            _normalize_article_id(document.article),
            _normalize_article_id(document.article_norm),
            _normalize_article_id(document.article_compact),
        } or expected_article in normalize_text(haystack or ""):
            score += 3
    if expected.series is not None:
        series_text = normalize_text(expected.series)
        document_series = infer_series_from_document(document)
        if series_text and document_series and series_text == document_series:
            score += 2
    return score


def infer_series_from_document(document: SteelProductDocument) -> str | None:
    explicit_series = getattr(document, "series", None)
    if explicit_series is not None:
        normalized = normalize_text(explicit_series)
        if normalized:
            return normalized
    haystack = " ".join(
        value for value in (document.name, document.article, document.article_norm) if value
    )
    normalized_haystack = normalize_text(haystack) or ""
    match = _SERIES_RE.search(normalized_haystack)
    if match:
        return match.group(1)
    return None


def score_preferred_documents(
    documents: Sequence[SteelProductDocument],
    expected: ExpectedAttributes,
) -> list[SteelProductDocument]:
    if not documents:
        return []
    if not has_soft_constraints(expected):
        return list(documents)

    scored = [(_soft_match_score(document, expected), document) for document in documents]
    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        return list(documents)
    return [document for score, document in scored if score == best_score]


def _document_articles(document: SteelProductDocument) -> list[str]:
    articles = [_normalize_article_id(document.article)]
    articles.extend(
        _normalize_article_id(candidate.article)
        for candidate in document.ld_candidates
        if candidate.article
    )
    return _normalize_article_list(articles)


def build_ld_articles_by_competitor(
    documents: Sequence[SteelProductDocument],
) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for document in documents:
        article = _normalize_article_id(document.article)
        if article not in mapping:
            mapping[article] = set()
        for candidate in document.ld_candidates:
            if candidate.article:
                mapping[article].add(_normalize_article_id(candidate.article))
    return {article: sorted(values) for article, values in sorted(mapping.items())}


def document_article(document: SteelProductDocument) -> str:
    return _normalize_article_id(document.article)


def document_articles(documents: Sequence[SteelProductDocument]) -> list[str]:
    return sorted({document_article(document) for document in documents})


def compact_number(value: float | None) -> str | None:
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def expected_from_document(document: SteelProductDocument) -> ExpectedAttributes:
    return ExpectedAttributes(
        brand=document.brand,
        dn=document.dn,
        pn_bar=document.pn_bar,
        connection=document.connection,
        body_material=document.body_material,
        medium=document.medium,
        control=document.control,
        temperature=document.temperature,
        length_mm=document.length_mm,
        series=None,
        article=None,
    )


def expected_to_extracted(expected: ExpectedAttributes) -> ExtractedAttributes:
    payload = expected.model_dump()
    payload.pop("brand", None)
    return ExtractedAttributes.model_validate(payload)


@dataclass(slots=True)
class ComparisonSummary:
    wrong_fields: list[str]
    hallucinated_fields: list[str]
    missing_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoldAttributeExtractor:
    """Deterministic extractor used for RAG-only and smoke testing."""

    def __init__(self, expected: ExpectedAttributes) -> None:
        self.expected = expected

    def extract(self, query: str) -> ExtractedAttributes:
        del query
        return expected_to_extracted(self.expected)


class GoldBrandDetector:
    """Deterministic brand detector used when DeepSeek is intentionally bypassed."""

    def __init__(self, brand: str | None) -> None:
        self.brand = brand

    def __call__(self, query: str) -> str | None:
        del query
        return self.brand
