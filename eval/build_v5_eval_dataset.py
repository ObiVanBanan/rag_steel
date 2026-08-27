"""Build the V5 evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eval.build_v4_eval_dataset import find_v4_eligible_documents
from eval.v3_common import (
    build_ld_articles_by_competitor,
    document_articles,
    score_preferred_documents,
)
from eval.v4_schema import EvalCase as V4EvalCase
from eval.v4_schema import ExpectedAttributes as V4ExpectedAttributes
from eval.v5_constants import (
    CATEGORY_ORDER,
    DEFAULT_GOLDEN_DATASET_PATH,
    DEFAULT_SOURCE_PATH,
    DEFAULT_SOURCE_PATHS,
)
from eval.v5_schema import EvalCase, ExpectedAttributes
from rag_steel.competitor_registry import COMPETITOR_BRANDS
from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.normalization import normalize_article, normalize_brand
from rag_steel.schemas import SteelProductDocument
from rag_steel.source_adapters import load_source_bundle

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


def _to_v5_expected(expected: V4ExpectedAttributes) -> ExpectedAttributes:
    brand = expected.resolved_brand if expected.raw_brand is not None else None
    return ExpectedAttributes(
        brand=brand,
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


def _build_expected(
    *,
    brand: str | None,
    resolved_brand: str | None,
    article: str | None,
    resolved_article: str | None = None,
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
        brand=brand,
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


def _make_case(
    *,
    category: str,
    query: str,
    expected_status: str,
    expected_resolution_mode: str,
    expected_attributes: ExpectedAttributes,
    source_documents: list[SteelProductDocument] | None = None,
    eligible_documents: list[SteelProductDocument] | None = None,
) -> EvalCase:
    if eligible_documents is None and source_documents is not None:
        eligible_documents = find_v4_eligible_documents(
            source_documents,
            resolved_brand=expected_attributes.resolved_brand,
            resolved_article=expected_attributes.resolved_article,
            dn=expected_attributes.dn,
            pn_bar=expected_attributes.pn_bar,
            connection=expected_attributes.connection,
        )
    eligible_documents = eligible_documents or []
    eligible_articles = document_articles(eligible_documents)
    preferred_documents = score_preferred_documents(eligible_documents, expected_attributes)
    preferred_articles = document_articles(preferred_documents)
    ld_mapping = build_ld_articles_by_competitor(eligible_documents)
    family_document = preferred_documents[0] if preferred_documents else None
    if family_document is None and eligible_documents:
        family_document = eligible_documents[0]
    return EvalCase(
        id="",
        category=category,
        query=query,
        expected_status=expected_status,
        expected_resolution_mode=expected_resolution_mode,
        product_family=(
            _product_family_for_document(family_document) if family_document is not None else None
        ),
        search_intent=_search_intent_for_category(category),
        expected_attributes=expected_attributes,
        eligible_competitor_articles=eligible_articles,
        preferred_competitor_articles=preferred_articles,
        expected_ld_articles_by_competitor=ld_mapping,
    )


def _supported_documents(documents: list[SteelProductDocument]) -> list[SteelProductDocument]:
    return [
        document
        for document in documents
        if document.brand and normalize_brand(document.brand) in SUPPORTED_BRANDS
    ]


def _product_family_for_document(document: SteelProductDocument | None) -> str | None:
    if document is None:
        return None
    name = (document.name or "").casefold()
    if "затвор" in name:
        return "butterfly_valve"
    if document.body_material and "латун" in document.body_material.casefold():
        return "brass_ball_valve"
    return "ball_valve"


def _search_intent_for_category(category: str) -> str:
    article_categories = {
        "article_only_exact",
        "article_only_normalized",
        "article_only_typo",
        "brand_plus_article",
        "article_plus_hard",
        "article_natural_language",
        "unknown_article",
        "ambiguous_article_typo",
        "brand_article_conflict",
        "article_hard_conflict",
        "v3_regression",
    }
    return "article" if category in article_categories else "description"


def _document_by_article_key(
    documents: list[SteelProductDocument],
) -> dict[str, SteelProductDocument]:
    return {_article_key(document): document for document in documents}


def _family_document_for_articles(
    article_values: Iterable[str],
    documents_by_article: dict[str, SteelProductDocument],
) -> SteelProductDocument | None:
    for article in article_values:
        document = documents_by_article.get(_compact_article(article))
        if document is not None:
            return document
    return None


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


def _load_v4_records(path: Path) -> list[V4EvalCase]:
    records: list[V4EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(V4EvalCase.model_validate_json(line))
    return records


def _transform_v4_records(
    records: list[V4EvalCase],
    documents: list[SteelProductDocument],
) -> list[EvalCase]:
    transformed: list[EvalCase] = []
    documents_by_article = _document_by_article_key(documents)
    for record in records:
        expected = record.expected_attributes
        query = record.query
        if query == "Temper DN999 PN999":
            query = "Temper DN107 PN999"
            expected = expected.model_copy(update={"dn": 100.0})
        resolved_article = expected.resolved_article or expected.article
        family_document = None
        if resolved_article:
            family_document = documents_by_article.get(_compact_article(resolved_article))
        if family_document is None:
            family_document = _family_document_for_articles(
                record.preferred_competitor_articles,
                documents_by_article,
            )
        if family_document is None:
            family_document = _family_document_for_articles(
                record.eligible_competitor_articles,
                documents_by_article,
            )
        transformed.append(
            EvalCase(
                id=record.id.replace("v4_", "v5_"),
                category=record.category,
                query=query,
                expected_status=record.expected_status,
                expected_resolution_mode=(
                    "brand_exact"
                    if record.category == "brand_typo"
                    else record.expected_resolution_mode
                ),
                product_family=(
                    _product_family_for_document(family_document)
                    if family_document is not None
                    else None
                ),
                search_intent=_search_intent_for_category(record.category),
                expected_attributes=_to_v5_expected(expected),
                eligible_competitor_articles=record.eligible_competitor_articles,
                preferred_competitor_articles=record.preferred_competitor_articles,
                expected_ld_articles_by_competitor=record.expected_ld_articles_by_competitor,
            )
        )
    return transformed


def _brand_query(document: SteelProductDocument, *, typo: bool = False) -> str:
    brand = normalize_brand(document.brand) or document.brand or "unknown"
    token = BRAND_TYPOS.get(brand, brand) if typo else brand
    parts = [
        token,
        f"DN{int(document.dn)}" if document.dn is not None else None,
        f"PN{int(document.pn_bar)}" if document.pn_bar is not None else None,
        document.connection,
    ]
    return " ".join(part for part in parts if part)


def _append_semantic_cases(
    records: list[EvalCase],
    documents: list[SteelProductDocument],
) -> None:
    brass_document = next(
        (
            document
            for document in documents
            if _product_family_for_document(document) == "brass_ball_valve"
        ),
        None,
    )
    butterfly_document = next(
        (
            document
            for document in documents
            if _product_family_for_document(document) == "butterfly_valve"
        ),
        None,
    )
    explicit_cases = [
        _make_case(
            category="brand_semantic",
            query="Темпер DN50",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Temper",
                resolved_brand="Temper",
                article=None,
                dn=50,
            ),
            source_documents=documents,
        ),
        _make_case(
            category="brand_semantic",
            query="Темпр DN50",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Temper",
                resolved_brand="Temper",
                article=None,
                dn=50,
            ),
            source_documents=documents,
        ),
        _make_case(
            category="brand_semantic",
            query="Броен DN50",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Broen",
                resolved_brand="Broen",
                article=None,
                dn=50,
            ),
            source_documents=documents,
        ),
        _make_case(
            category="brand_semantic",
            query="Маршал сотка",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="MARSHAL",
                resolved_brand="MARSHAL",
                article=None,
                dn=100,
            ),
            source_documents=documents,
        ),
        _make_case(
            category="unsupported_brand",
            query="NoSuchBrand DN50",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=50,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="dn_semantic",
            query="DN51",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=50,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="dn_semantic",
            query="ду 64",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=65,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="dn_semantic",
            query="сотка",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=100,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="dn_semantic",
            query="пятидесятый",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=50,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="ambiguous_semantic",
            query="DN57",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=50,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="pn_semantic",
            query="1.6 МПа",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                pn_bar=16,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="pn_semantic",
            query="ру шестнадцать",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                pn_bar=16,
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="connection_semantic",
            query="на фланцах",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                connection="фланцевое",
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="connection_semantic",
            query="под сварку",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                connection="сварное",
            ),
            eligible_documents=[],
        ),
        _make_case(
            category="mixed_semantic",
            query="темпер ду51 ру16 фланец",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Temper",
                resolved_brand="Temper",
                article=None,
                dn=50,
                pn_bar=16,
                connection="фланцевое",
            ),
            source_documents=documents,
        ),
        _make_case(
            category="mixed_semantic",
            query="нужен броен сотка на 25 под сварку",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Broen",
                resolved_brand="Broen",
                article=None,
                dn=100,
                pn_bar=25,
                connection="сварное",
            ),
            source_documents=documents,
        ),
        _make_case(
            category="real_user_regression",
            query="Нужен аналог крана Temper Ду 20 фланцевого Ру 40",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Temper",
                resolved_brand="Temper",
                article=None,
                dn=20,
                pn_bar=40,
                connection="фланцевое",
            ),
            source_documents=documents,
        ),
        _make_case(
            category="real_user_regression",
            query="Нужен темпер ду20 ру40 на фланцах",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Temper",
                resolved_brand="Temper",
                article=None,
                dn=20,
                pn_bar=40,
                connection="фланцевое",
            ),
            source_documents=documents,
        ),
        _make_case(
            category="real_user_regression",
            query="броен сотка 16 бар",
            expected_status="exact_match",
            expected_resolution_mode="brand_exact",
            expected_attributes=_build_expected(
                brand="Broen",
                resolved_brand="Broen",
                article=None,
                dn=100,
                pn_bar=16,
            ),
            source_documents=documents,
        ),
        _make_case(
            category="ambiguous_semantic",
            query="какой-то кран 18 бар",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                pn_bar=18,
            ),
            eligible_documents=[],
        ),
    ]
    if brass_document is not None:
        brass_brand = normalize_brand(brass_document.brand) or brass_document.brand
        explicit_cases.append(
            _make_case(
                category="mixed_semantic",
                query=(
                    f"Нужен латунный кран {brass_brand} "
                    f"DN{int(brass_document.dn)} PN{int(brass_document.pn_bar)} резьбовое"
                ),
                expected_status="exact_match",
                expected_resolution_mode="brand_exact",
                expected_attributes=_build_expected(
                    brand=brass_brand,
                    resolved_brand=brass_brand,
                    article=None,
                    dn=brass_document.dn,
                    pn_bar=brass_document.pn_bar,
                    connection=brass_document.connection,
                    body_material=brass_document.body_material,
                ),
                source_documents=documents,
            )
        )
    if butterfly_document is not None:
        butterfly_brand = normalize_brand(butterfly_document.brand) or butterfly_document.brand
        explicit_cases.append(
            _make_case(
                category="connection_semantic",
                query=(
                    f"Подбери затвор {butterfly_brand} "
                    f"DN{int(butterfly_document.dn)} PN{int(butterfly_document.pn_bar)} "
                    f"{butterfly_document.connection}"
                ),
                expected_status="exact_match",
                expected_resolution_mode="brand_exact",
                expected_attributes=_build_expected(
                    brand=butterfly_brand,
                    resolved_brand=butterfly_brand,
                    article=None,
                    dn=butterfly_document.dn,
                    pn_bar=butterfly_document.pn_bar,
                    connection=butterfly_document.connection,
                ),
                source_documents=documents,
            )
        )

    records.extend(explicit_cases)


def build_v5_cases(
    documents: list[SteelProductDocument],
    *,
    target_count: int = 180,
) -> tuple[list[EvalCase], dict[str, Any]]:
    source_brand_counts = Counter(
        normalize_brand(document.brand) if document.brand else "none" for document in documents
    )
    documents = _supported_documents(documents)
    v4_path = Path("eval/v4_golden_queries.jsonl")
    if not v4_path.exists():
        from eval.build_v4_eval_dataset import build_v4_dataset

        build_v4_dataset(output_path=v4_path)

    base_records = _load_v4_records(v4_path)
    records = _transform_v4_records(base_records, documents)
    _append_semantic_cases(records, documents)

    records.sort(
        key=lambda item: (
            CATEGORY_ORDER.index(item.category) if item.category in CATEGORY_ORDER else 999,
            item.query,
        )
    )
    if len(records) > target_count:
        records = records[:target_count]

    for index, record in enumerate(records, start=1):
        record.id = f"v5_{record.category}_{index:04d}"

    semantic_cases = sum(1 for record in records if record.category.endswith("semantic"))
    negative_cases = sum(1 for record in records if record.expected_status != "exact_match")
    brand_cases = sum(1 for record in records if record.expected_attributes.brand is not None)

    meta = {
        "source_documents": len(documents),
        "records": len(records),
        "by_category": dict(Counter(record.category for record in records)),
        "by_brand": dict(
            Counter(
                record.expected_attributes.resolved_brand
                or record.expected_attributes.brand
                or "none"
                for record in records
            )
        ),
        "semantic_cases": semantic_cases,
        "article_cases": sum(1 for record in records if record.search_intent == "article"),
        "description_cases": sum(1 for record in records if record.search_intent == "description"),
        "negative_cases": negative_cases,
        "brand_cases": brand_cases,
        "by_product_family": dict(
            Counter(record.product_family or "unknown" for record in records)
        ),
        "source_brand_counts": dict(source_brand_counts),
        "notes": [],
    }
    return records, meta


def _write_jsonl(records: Iterable[EvalCase], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return output_path


def build_v5_dataset(
    *,
    csv_path: Path = DEFAULT_SOURCE_PATH,
    csv_paths: list[Path] | tuple[Path, ...] | None = None,
    output_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    target_count: int = 180,
) -> tuple[list[EvalCase], dict[str, Any]]:
    if csv_paths is not None:
        frame, source_files = load_source_bundle(list(csv_paths))
    else:
        default_paths = [path for path in DEFAULT_SOURCE_PATHS if path.exists()]
        if len(default_paths) > 1:
            frame, source_files = load_source_bundle(default_paths)
        else:
            frame = pd.read_csv(csv_path)
            source_files = []
    documents = build_source_documents_from_frame(frame)
    records, meta = build_v5_cases(documents, target_count=target_count)
    if source_files:
        meta["source_files"] = [record.to_dict() for record in source_files]
    _write_jsonl(records, output_path)
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 evaluation datasets.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH, help="Source CSV path")
    parser.add_argument(
        "--csv-bundle",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional source bundle paths; when omitted the default multisource bundle "
            "is used if present"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GOLDEN_DATASET_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=180,
        help="Number of V5 cases to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    records, meta = build_v5_dataset(
        csv_path=args.csv,
        csv_paths=args.csv_bundle,
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


__all__ = ["build_v5_cases", "build_v5_dataset"]


if __name__ == "__main__":
    raise SystemExit(main())
