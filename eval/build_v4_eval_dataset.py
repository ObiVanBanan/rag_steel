"""Build the V4 evaluation dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eval.v3_common import build_ld_articles_by_competitor, document_articles
from eval.v4_constants import CATEGORY_ORDER, DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_SOURCE_PATH
from eval.v4_schema import EvalCase, ExpectedAttributes
from rag_steel.competitor_registry import COMPETITOR_BRANDS
from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.normalization import normalize_article, normalize_brand
from rag_steel.schemas import SteelProductDocument

SUPPORTED_BRANDS = tuple(COMPETITOR_BRANDS)
BRAND_TYPOS = {
    "Temper": "Tempr",
    "ALSO": "ALSOO",
    "MARSHAL": "Marsha",
    "Broen": "Broem",
    "FORTECA": "Fortecaa",
    "Бивал": "Бивл",
}
EXACT_ONLY_BRANDS = {"ADL"}


def _article_key(document: SteelProductDocument) -> str:
    normalized = normalize_article(document.article)
    return (normalized.article_compact or normalized.article_norm or document.article).casefold()


def _compact_article(value: str) -> str:
    normalized = normalize_article(value)
    return normalized.article_compact or normalized.article_norm or value.casefold()


def _distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diffs = [
            index for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)) if lhs != rhs
        ]
        if len(diffs) == 1:
            return True
        if (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and left[diffs[0]] == right[diffs[1]]
            and left[diffs[1]] == right[diffs[0]]
        ):
            return True
        return False
    if len(left) > len(right):
        left, right = right, left
    i = j = 0
    edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    if i < len(left) or j < len(right):
        edits += 1
    return edits <= 1


def _build_expected(
    *,
    raw_brand: str | None,
    resolved_brand: str | None,
    article: str | None,
    resolved_article: str | None,
    dn: float | None = None,
    pn_bar: float | None = None,
    connection: str | None = None,
    body_material: str | None = None,
    medium: str | None = None,
    control: str | None = None,
    temperature: str | None = None,
    length_mm: float | None = None,
    series: str | None = None,
) -> ExpectedAttributes:
    return ExpectedAttributes(
        raw_brand=raw_brand,
        resolved_brand=resolved_brand,
        article=article,
        resolved_article=resolved_article,
        dn=dn,
        pn_bar=pn_bar,
        connection=connection,
        body_material=body_material,
        medium=medium,
        control=control,
        temperature=temperature,
        length_mm=length_mm,
        series=series,
    )


def _source_brand_counts(documents: Iterable[SteelProductDocument]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for document in documents:
        brand = normalize_brand(document.brand) if document.brand else None
        counts[brand or "none"] += 1
    return counts


def _make_case(
    *,
    category: str,
    query: str,
    expected_status: str,
    expected_resolution_mode: str,
    expected_attributes: ExpectedAttributes,
    eligible_documents: list[SteelProductDocument] | None = None,
) -> EvalCase:
    eligible_documents = eligible_documents or []
    eligible_articles = document_articles(eligible_documents)
    preferred_articles = eligible_articles
    ld_mapping = build_ld_articles_by_competitor(eligible_documents)
    return EvalCase(
        id="",
        category=category,
        query=query,
        expected_status=expected_status,
        expected_resolution_mode=expected_resolution_mode,
        expected_attributes=expected_attributes,
        eligible_competitor_articles=eligible_articles,
        preferred_competitor_articles=preferred_articles,
        expected_ld_articles_by_competitor=ld_mapping,
    )


def _select_balanced_documents(
    documents: list[SteelProductDocument],
    *,
    count: int,
) -> list[SteelProductDocument]:
    grouped: dict[str, list[SteelProductDocument]] = defaultdict(list)
    for document in documents:
        if not document.brand:
            continue
        brand = normalize_brand(document.brand)
        if brand in SUPPORTED_BRANDS:
            grouped[brand].append(document)

    ordered_brands = [brand for brand in SUPPORTED_BRANDS if brand in grouped]
    for brand in ordered_brands:
        grouped[brand].sort(key=lambda item: _article_key(item))

    result: list[SteelProductDocument] = []
    for _, brand in zip(range(count), cycle(ordered_brands), strict=False):
        bucket = grouped[brand]
        if not bucket:
            continue
        result.append(bucket.pop(0))
        if len(result) >= count:
            break
    if len(result) < count:
        remaining = [doc for brand in ordered_brands for doc in grouped[brand]]
        remaining.sort(key=lambda item: (normalize_brand(item.brand) or "", _article_key(item)))
        for document in remaining:
            result.append(document)
            if len(result) >= count:
                break
    return result


def _exact_article_query(document: SteelProductDocument) -> str:
    return document.article


def _normalized_article_query(document: SteelProductDocument, ordinal: int) -> str:
    article = document.article
    normalized = normalize_article(article)
    variants = [
        article.lower(),
        article.upper(),
        normalized.article_norm or article,
        normalized.article_compact or article,
        re.sub(r"[./_-]+", " ", article).strip(),
        f"{article[: max(1, len(article) // 2)]} {article[max(1, len(article) // 2) :]}".strip(),
    ]
    variant = variants[ordinal % len(variants)]
    return variant if variant != article else variants[(ordinal + 1) % len(variants)]


def _all_article_keys(documents: Iterable[SteelProductDocument]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for document in documents:
        key = _article_key(document)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _unique_article_typo(
    document: SteelProductDocument,
    catalog_keys: list[str],
) -> str | None:
    compact = _compact_article(document.article)
    if len(compact) < 2:
        return None

    substitution_chars = "0123456789abcxyzабвгдеёжзиклмнопрстуфхцчшщ"

    candidates: list[str] = []
    for index, char in enumerate(compact):
        for replacement in substitution_chars:
            if replacement == char:
                continue
            candidates.append(compact[:index] + replacement + compact[index + 1 :])
        candidates.append(compact[:index] + compact[index + 1 :])
        candidates.append(compact[:index] + char + compact[index:])
        if index + 1 < len(compact):
            candidates.append(
                compact[:index] + compact[index + 1] + compact[index] + compact[index + 2 :]
            )

    for candidate in candidates:
        matches = [key for key in catalog_keys if _distance_at_most_one(candidate, key)]
        if len(matches) == 1:
            return candidate
    return None


def _ambiguous_article_typo_pair(
    documents: list[SteelProductDocument],
) -> tuple[str, SteelProductDocument, SteelProductDocument] | None:
    signature_groups: dict[tuple[int, str, str], list[SteelProductDocument]] = defaultdict(list)
    for document in documents:
        compact = _compact_article(document.article)
        if len(compact) < 2:
            continue
        for index in range(len(compact)):
            signature_groups[(index, compact[:index], compact[index + 1 :])].append(document)

    for (index, prefix, suffix), group in signature_groups.items():
        if len(group) < 2:
            continue
        chars = {_compact_article(doc.article)[index] for doc in group}
        replacement = next((char for char in "7xyzабв" if char not in chars), None)
        if replacement is None:
            continue
        query = prefix + replacement + suffix
        return query, group[0], group[1]
    return None


def _supported_documents(documents: list[SteelProductDocument]) -> list[SteelProductDocument]:
    return [
        document
        for document in documents
        if document.brand and normalize_brand(document.brand) in SUPPORTED_BRANDS
    ]


def _pn_minimum_query(document: SteelProductDocument) -> str:
    parts = [
        document.brand,
        f"DN{int(document.dn)}" if document.dn is not None else None,
        f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
    ]
    return " ".join(part for part in parts if part)


def _pn_minimum_expected(document: SteelProductDocument) -> ExpectedAttributes:
    return ExpectedAttributes(
        raw_brand=document.brand,
        resolved_brand=document.brand,
        article=None,
        resolved_article=None,
        dn=document.dn,
        pn_bar=document.pn_bar,
    )


def _pn_minimum_semantics_cases(
    documents: list[SteelProductDocument],
    *,
    count: int = 10,
) -> list[EvalCase]:
    grouped: dict[tuple[str, float], list[SteelProductDocument]] = defaultdict(list)
    for document in documents:
        if not document.brand or document.dn is None or document.pn_bar is None:
            continue
        grouped[(normalize_brand(document.brand) or document.brand, float(document.dn))].append(
            document
        )

    candidates: list[
        tuple[str, float, float, str, SteelProductDocument, list[SteelProductDocument]]
    ] = []
    for (brand, dn), group in grouped.items():
        ordered = sorted(group, key=lambda item: (float(item.pn_bar), _article_key(item)))
        pn_values = sorted({float(item.pn_bar) for item in ordered})
        if len(pn_values) < 3:
            continue
        for document in ordered:
            requested_pn = float(document.pn_bar)
            lower_exists = any(float(item.pn_bar) < requested_pn for item in ordered)
            higher_exists = any(float(item.pn_bar) > requested_pn for item in ordered)
            if not lower_exists or not higher_exists:
                continue
            eligible_documents = [
                item for item in ordered if float(item.pn_bar) >= requested_pn
            ]
            if len(eligible_documents) < 2:
                continue
            candidates.append(
                (
                    brand,
                    dn,
                    requested_pn,
                    _article_key(document),
                    document,
                    eligible_documents,
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    records: list[EvalCase] = []
    for _brand, _dn, _requested_pn, _article_key_value, document, eligible_documents in candidates:
        records.append(
            _make_case(
                category="pn_minimum_semantics",
                query=_pn_minimum_query(document),
                expected_status="exact_match",
                expected_resolution_mode="brand_exact",
                expected_attributes=_pn_minimum_expected(document),
                eligible_documents=eligible_documents,
            )
        )
        if len(records) >= count:
            break
    return records


def _regression_cases(documents: list[SteelProductDocument], target_count: int) -> list[EvalCase]:
    from eval.build_v3_eval_dataset import build_v3_cases

    records, _meta = build_v3_cases(documents, target_count=target_count)
    converted: list[EvalCase] = []
    for record in records:
        expected = record.expected_attributes
        expected_resolution_mode = (
            "no_identity"
            if expected.brand is None
            else "brand_and_article"
            if expected.article is not None
            else "brand_exact"
        )
        converted.append(
            EvalCase(
                id=record.id.replace("v3_", "v4_"),
                category="v3_regression",
                query=record.query,
                expected_status=record.expected_status,
                expected_resolution_mode=expected_resolution_mode,
                expected_attributes=ExpectedAttributes(
                    raw_brand=expected.brand,
                    resolved_brand=expected.brand,
                    article=expected.article,
                    resolved_article=expected.article,
                    dn=expected.dn,
                    pn_bar=expected.pn_bar,
                    connection=expected.connection,
                    body_material=expected.body_material,
                    medium=expected.medium,
                    control=expected.control,
                    temperature=expected.temperature,
                    length_mm=expected.length_mm,
                    series=expected.series,
                ),
                eligible_competitor_articles=record.eligible_competitor_articles,
                preferred_competitor_articles=record.preferred_competitor_articles,
                expected_ld_articles_by_competitor=record.expected_ld_articles_by_competitor,
            )
        )
    return converted


def build_v4_cases(
    documents: list[SteelProductDocument],
    *,
    target_count: int = 160,
) -> tuple[list[EvalCase], dict[str, Any]]:
    source_brand_counts = _source_brand_counts(documents)
    documents = _supported_documents(documents)
    catalog_keys = _all_article_keys(documents)
    exact_docs = _select_balanced_documents(documents, count=15)
    normalized_docs = _select_balanced_documents(documents[15:] + documents[:15], count=10)
    typo_docs = _select_balanced_documents(documents[25:] + documents[:25], count=10)
    brand_docs = _select_balanced_documents(documents[35:] + documents[:35], count=10)
    brand_article_docs = _select_balanced_documents(documents[45:] + documents[:45], count=10)
    hard_docs = _select_balanced_documents(documents[55:] + documents[:55], count=10)
    natural_docs = _select_balanced_documents(documents[65:] + documents[:65], count=10)
    unknown_docs = _select_balanced_documents(documents[75:] + documents[:75], count=8)
    conflict_docs = _select_balanced_documents(documents[83:] + documents[:83], count=6)

    records: list[EvalCase] = []
    counts: Counter[str] = Counter()
    by_brand: Counter[str] = Counter()

    for _index, document in enumerate(exact_docs):
        records.append(
            _make_case(
                category="article_only_exact",
                query=_exact_article_query(document),
                expected_status="exact_match",
                expected_resolution_mode="article_exact",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=document.brand,
                    article=document.article,
                    resolved_article=document.article,
                ),
                eligible_documents=[document],
            )
        )
        counts["article_only_exact"] += 1
        by_brand[document.brand or "none"] += 1

    for _index, document in enumerate(normalized_docs):
        records.append(
            _make_case(
                category="article_only_normalized",
                query=_normalized_article_query(document, _index),
                expected_status="exact_match",
                expected_resolution_mode="article_exact",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=document.brand,
                    article=document.article,
                    resolved_article=document.article,
                ),
                eligible_documents=[document],
            )
        )
        counts["article_only_normalized"] += 1
        by_brand[document.brand or "none"] += 1

    for document in typo_docs:
        typo = _unique_article_typo(document, catalog_keys)
        if typo is None:
            continue
        records.append(
            _make_case(
                category="article_only_typo",
                query=typo,
                expected_status="exact_match",
                expected_resolution_mode="article_fuzzy",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=document.brand,
                    article=typo,
                    resolved_article=document.article,
                ),
                eligible_documents=[document],
            )
        )
        counts["article_only_typo"] += 1
        by_brand[document.brand or "none"] += 1

    for _index, document in enumerate(brand_docs):
        brand = document.brand or "unknown"
        if brand in EXACT_ONLY_BRANDS:
            continue
        typo = BRAND_TYPOS.get(brand, brand)
        query = " ".join(
            part
            for part in (
                typo,
                f"DN{int(document.dn)}" if document.dn is not None else None,
                f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
                document.connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="brand_typo",
                query=query,
                expected_status="exact_match",
                expected_resolution_mode="brand_fuzzy",
                expected_attributes=_build_expected(
                    raw_brand=typo,
                    resolved_brand=document.brand,
                    article=None,
                    resolved_article=None,
                    dn=document.dn,
                    pn_bar=document.pn_bar,
                    connection=document.connection,
                ),
                eligible_documents=[document],
            )
        )
        counts["brand_typo"] += 1
        by_brand[document.brand or "none"] += 1

    for document in brand_article_docs:
        query = " ".join(
            part
            for part in (
                document.brand,
                document.article,
                f"DN{int(document.dn)}" if document.dn is not None else None,
                f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
                document.connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="brand_plus_article",
                query=query,
                expected_status="exact_match",
                expected_resolution_mode="brand_and_article",
                expected_attributes=_build_expected(
                    raw_brand=document.brand,
                    resolved_brand=document.brand,
                    article=document.article,
                    resolved_article=document.article,
                    dn=document.dn,
                    pn_bar=document.pn_bar,
                    connection=document.connection,
                ),
                eligible_documents=[document],
            )
        )
        counts["brand_plus_article"] += 1
        by_brand[document.brand or "none"] += 1

    for document in hard_docs:
        query = " ".join(
            part
            for part in (
                document.article,
                f"DN{int(document.dn)}" if document.dn is not None else None,
                f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
                document.connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="article_plus_hard",
                query=query,
                expected_status="exact_match",
                expected_resolution_mode="article_exact",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=document.brand,
                    article=document.article,
                    resolved_article=document.article,
                    dn=document.dn,
                    pn_bar=document.pn_bar,
                    connection=document.connection,
                ),
                eligible_documents=[document],
            )
        )
        counts["article_plus_hard"] += 1
        by_brand[document.brand or "none"] += 1

    for document in natural_docs:
        query = " ".join(
            part
            for part in (
                "Нужен аналог для",
                document.article,
                document.brand,
                f"DN{int(document.dn)}" if document.dn is not None else None,
                f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
                document.connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="article_natural_language",
                query=query,
                expected_status="exact_match",
                expected_resolution_mode="article_exact",
                expected_attributes=_build_expected(
                    raw_brand=document.brand,
                    resolved_brand=document.brand,
                    article=document.article,
                    resolved_article=document.article,
                    dn=document.dn,
                    pn_bar=document.pn_bar,
                    connection=document.connection,
                ),
                eligible_documents=[document],
            )
        )
        counts["article_natural_language"] += 1
        by_brand[document.brand or "none"] += 1

    for document in unknown_docs:
        query = f"ZZ-{_article_key(document)[:8]}-UNKNOWN"
        records.append(
            _make_case(
                category="unknown_article",
                query=query,
                expected_status="not_found",
                expected_resolution_mode="article_not_found",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=None,
                    article=query,
                    resolved_article=None,
                ),
            )
        )
        counts["unknown_article"] += 1
        by_brand["none"] += 1

    ambiguous = _ambiguous_article_typo_pair(documents)
    if ambiguous is not None:
        query, first, second = ambiguous
        records.append(
            _make_case(
                category="ambiguous_article_typo",
                query=query,
                expected_status="not_found",
                expected_resolution_mode="article_ambiguous",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=None,
                    article=query,
                    resolved_article=None,
                ),
                eligible_documents=[first, second],
            )
        )
        counts["ambiguous_article_typo"] += 1
        by_brand[first.brand or "none"] += 1

    for _index, document in enumerate(conflict_docs[:3]):
        wrong_brand = next(
            brand
            for brand in SUPPORTED_BRANDS
            if normalize_brand(document.brand) != normalize_brand(brand)
        )
        query = " ".join(
            part
            for part in (
                wrong_brand,
                document.article,
                f"DN{int(document.dn)}" if document.dn is not None else None,
                f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
                document.connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="brand_article_conflict",
                query=query,
                expected_status="not_found",
                expected_resolution_mode="identity_conflict",
                expected_attributes=_build_expected(
                    raw_brand=wrong_brand,
                    resolved_brand=wrong_brand,
                    article=document.article,
                    resolved_article=None,
                    dn=document.dn,
                    pn_bar=document.pn_bar,
                    connection=document.connection,
                ),
            )
        )
        counts["brand_article_conflict"] += 1
        by_brand[document.brand or "none"] += 1

    for _index, document in enumerate(conflict_docs[3:6]):
        conflict_dn = float((document.dn or 0) + 7) if document.dn is not None else None
        conflict_pn = float((document.pn_bar or 0) + 3) if document.pn_bar is not None else None
        conflict_connection = (
            "фланцевое" if document.connection != "фланцевое" else "резьбовое"
        )
        query = " ".join(
            part
            for part in (
                document.article,
                f"DN{int(conflict_dn)}" if conflict_dn is not None else None,
                f"PN{int(conflict_pn)}" if conflict_pn is not None else None,
                conflict_connection,
            )
            if part
        )
        records.append(
            _make_case(
                category="article_hard_conflict",
                query=query,
                expected_status="not_found",
                expected_resolution_mode="identity_conflict",
                expected_attributes=_build_expected(
                    raw_brand=None,
                    resolved_brand=None,
                    article=document.article,
                    resolved_article=None,
                    dn=conflict_dn,
                    pn_bar=conflict_pn,
                    connection=conflict_connection,
                ),
            )
        )
        counts["article_hard_conflict"] += 1
        by_brand[document.brand or "none"] += 1

    pn_cases = _pn_minimum_semantics_cases(documents, count=10)
    records.extend(pn_cases)
    counts["pn_minimum_semantics"] += len(pn_cases)
    for case in pn_cases:
        by_brand[
            case.expected_attributes.resolved_brand
            or case.expected_attributes.raw_brand
            or "none"
        ] += 1

    regression_cases = _regression_cases(documents, target_count=60)
    records.extend(regression_cases)
    counts["v3_regression"] += len(regression_cases)
    for case in regression_cases:
        by_brand[
            case.expected_attributes.resolved_brand
            or case.expected_attributes.raw_brand
            or "none"
        ] += 1

    records.sort(
        key=lambda item: (
            CATEGORY_ORDER.index(item.category) if item.category in CATEGORY_ORDER else 999,
            item.query,
        )
    )
    if len(records) > target_count:
        records = records[:target_count]

    for index, record in enumerate(records, start=1):
        record.id = f"v4_{record.category}_{index:04d}"

    article_cases = sum(
        1
        for record in records
        if record.category.startswith("article_")
        or record.category in {"unknown_article", "ambiguous_article_typo"}
    )
    typo_cases = sum(1 for record in records if "typo" in record.category)
    negative_cases = sum(1 for record in records if record.expected_status != "exact_match")
    adl_cases = sum(
        1 for record in records if (record.expected_attributes.raw_brand or "").upper() == "ADL"
    )

    meta = {
        "source_documents": len(documents),
        "records": len(records),
        "by_category": dict(Counter(record.category for record in records)),
        "by_brand": dict(
            Counter(
                record.expected_attributes.resolved_brand
                or record.expected_attributes.raw_brand
                or "none"
                for record in records
            )
        ),
        "article_cases": article_cases,
        "typo_cases": typo_cases,
        "negative_cases": negative_cases,
        "adl_cases": adl_cases,
        "source_brand_counts": dict(source_brand_counts),
        "adl_available": source_brand_counts.get("ADL", 0) > 0,
        "notes": [],
    }
    if not meta["adl_available"]:
        meta["notes"].append(
            "ADL is absent from the source dataset, so ADL-specific cases were not generated."
        )
    return records, meta


def _write_jsonl(records: Iterable[EvalCase], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return output_path


def build_v4_dataset(
    *,
    csv_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    target_count: int = 160,
) -> tuple[list[EvalCase], dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    documents = build_source_documents_from_frame(frame)
    records, meta = build_v4_cases(documents, target_count=target_count)
    _write_jsonl(records, output_path)
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V4 evaluation datasets.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH, help="Source CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GOLDEN_DATASET_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=160,
        help="Number of V4 cases to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    records, meta = build_v4_dataset(
        csv_path=args.csv,
        output_path=args.output,
        target_count=args.target_count,
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "records": len(records),
                "meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = ["build_v4_cases", "build_v4_dataset"]


if __name__ == "__main__":
    raise SystemExit(main())
