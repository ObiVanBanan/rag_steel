"""Build V3 golden and generated evaluation datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eval.v3_common import (
    build_ld_articles_by_competitor,
    compact_number,
    document_articles,
    find_hard_eligible_documents,
    infer_series_from_document,
    score_preferred_documents,
)
from eval.v3_constants import (
    CATEGORY_ORDER,
    DEFAULT_GENERATED_DATASET_PATH,
    DEFAULT_GOLDEN_DATASET_PATH,
    DEFAULT_SOURCE_PATH,
)
from eval.v3_schema import EvalCase, ExpectedAttributes
from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.normalization import normalize_brand
from rag_steel.schemas import SteelProductDocument

ALIAS_BY_CANONICAL = {
    "Temper": ("temper",),
    "ALSO": ("also", "алсо"),
    "MARSHAL": ("marshal", "маршал"),
    "Broen": ("broen", "брон"),
    "ADL": ("adl", "адл"),
    "FORTECA": ("forteca", "фортека"),
    "Бивал": ("бивал", "bival"),
}


def _field_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _expected_only(
    document: SteelProductDocument,
    *fields: str,
    overrides: dict[str, Any] | None = None,
) -> ExpectedAttributes:
    data = {field: None for field in ExpectedAttributes.model_fields}
    for field in fields:
        if field == "series":
            data[field] = infer_series_from_document(document)
        else:
            data[field] = getattr(document, field)
    if overrides:
        data.update(overrides)
    return ExpectedAttributes.model_validate(data)


def _query_hard(document: SteelProductDocument) -> str:
    parts = [
        document.brand,
        f"DN{compact_number(document.dn)}" if document.dn is not None else None,
        f"PN{compact_number(document.pn_bar)}" if document.pn_bar is not None else None,
        document.connection,
    ]
    return " ".join(part for part in parts if part)


def _query_hard_plus_material(document: SteelProductDocument) -> str | None:
    if document.body_material is None:
        return None
    return f"{_query_hard(document)} {document.body_material}"


def _query_hard_plus_medium(document: SteelProductDocument) -> str | None:
    if document.medium is None:
        return None
    return f"{_query_hard(document)} {document.medium}"


def _query_hard_plus_series(document: SteelProductDocument) -> str | None:
    series = infer_series_from_document(document)
    if series is None:
        return None
    return (
        f"{document.brand} series {series} "
        f"DN{compact_number(document.dn)} PN{compact_number(document.pn_bar)}"
    )


def _query_hard_plus_article(document: SteelProductDocument) -> str:
    return f"{_query_hard(document)} {document.article}"


def _query_hard_plus_multiple_soft(document: SteelProductDocument) -> str | None:
    soft_parts = [
        document.body_material,
        document.medium,
        document.control,
        document.temperature,
    ]
    parts = [part for part in soft_parts if part]
    if len(parts) < 2:
        return None
    return f"{_query_hard(document)} {' '.join(parts)}"


def _query_missing_dn(document: SteelProductDocument) -> str | None:
    if document.pn_bar is None:
        return None
    parts = [
        document.brand,
        f"PN{compact_number(document.pn_bar)}",
        document.connection,
    ]
    return " ".join(part for part in parts if part)


def _query_missing_pn(document: SteelProductDocument) -> str | None:
    if document.dn is None:
        return None
    parts = [
        document.brand,
        f"DN{compact_number(document.dn)}",
        document.connection,
    ]
    return " ".join(part for part in parts if part)


def _query_missing_connection(document: SteelProductDocument) -> str | None:
    if document.dn is None or document.pn_bar is None:
        return None
    parts = [
        document.brand,
        f"DN{compact_number(document.dn)}",
        f"PN{compact_number(document.pn_bar)}",
    ]
    return " ".join(part for part in parts if part)


def _query_russian_alias(document: SteelProductDocument) -> str | None:
    aliases = ALIAS_BY_CANONICAL.get(document.brand or "", ())
    if not aliases:
        return None
    parts = [
        aliases[0],
        f"Ду{compact_number(document.dn)}" if document.dn is not None else None,
        f"Ру{compact_number(document.pn_bar)}" if document.pn_bar is not None else None,
    ]
    return " ".join(part for part in parts if part)


def _query_compact_syntax(document: SteelProductDocument) -> str | None:
    if document.dn is None or document.pn_bar is None:
        return None
    return f"{document.brand} DN{compact_number(document.dn)}PN{compact_number(document.pn_bar)}"


def _query_natural_language(document: SteelProductDocument) -> str | None:
    if document.dn is None or document.pn_bar is None:
        return None
    parts = [
        "Подбери мне аналог",
        document.brand,
        f"на диаметр {compact_number(document.dn)}",
        f"давление {compact_number(document.pn_bar)} бар",
        document.connection,
        document.body_material,
        document.medium,
    ]
    return " ".join(part for part in parts if part)


def _query_no_brand(document: SteelProductDocument) -> str | None:
    if document.dn is None or document.pn_bar is None:
        return None
    parts = [
        "Нужен шаровой кран",
        f"DN{compact_number(document.dn)}",
        f"PN{compact_number(document.pn_bar)}",
        document.connection,
    ]
    return " ".join(part for part in parts if part)


def _query_impossible_hard(document: SteelProductDocument) -> str | None:
    if document.dn is None or document.pn_bar is None or not document.brand:
        return None
    return f"{document.brand} DN999 PN999"


def _expected_for_query(
    document: SteelProductDocument,
    *,
    category: str,
) -> ExpectedAttributes:
    if category == "hard_only":
        return _expected_only(document, "brand", "dn", "pn_bar", "connection")
    if category == "hard_plus_material":
        return _expected_only(document, "brand", "dn", "pn_bar", "connection", "body_material")
    if category == "hard_plus_medium":
        return _expected_only(document, "brand", "dn", "pn_bar", "connection", "medium")
    if category == "hard_plus_series":
        return _expected_only(document, "brand", "dn", "pn_bar", "series")
    if category == "hard_plus_article":
        return _expected_only(document, "brand", "dn", "pn_bar", "connection", "article")
    if category == "hard_plus_multiple_soft":
        return _expected_only(
            document,
            "brand",
            "dn",
            "pn_bar",
            "connection",
            "body_material",
            "medium",
            "control",
            "temperature",
        )
    if category == "missing_dn":
        return _expected_only(document, "brand", "pn_bar", "connection")
    if category == "missing_pn":
        return _expected_only(document, "brand", "dn", "connection")
    if category == "missing_connection":
        return _expected_only(document, "brand", "dn", "pn_bar")
    if category == "russian_alias":
        return _expected_only(document, "brand", "dn", "pn_bar")
    if category == "compact_syntax":
        return _expected_only(document, "brand", "dn", "pn_bar")
    if category == "natural_language":
        return _expected_only(
            document, "brand", "dn", "pn_bar", "connection", "body_material", "medium"
        )
    if category == "no_brand":
        return _expected_only(
            document,
            "dn",
            "pn_bar",
            "connection",
            overrides={"brand": None},
        )
    if category == "impossible_hard":
        return _expected_only(
            document,
            "brand",
            overrides={"dn": 999, "pn_bar": 999},
        )
    raise ValueError(f"Unsupported category: {category}")


def _query_for_category(document: SteelProductDocument, category: str) -> str | None:
    builders = {
        "hard_only": _query_hard,
        "hard_plus_material": _query_hard_plus_material,
        "hard_plus_medium": _query_hard_plus_medium,
        "hard_plus_series": _query_hard_plus_series,
        "hard_plus_article": _query_hard_plus_article,
        "hard_plus_multiple_soft": _query_hard_plus_multiple_soft,
        "missing_dn": _query_missing_dn,
        "missing_pn": _query_missing_pn,
        "missing_connection": _query_missing_connection,
        "russian_alias": _query_russian_alias,
        "compact_syntax": _query_compact_syntax,
        "natural_language": _query_natural_language,
        "no_brand": _query_no_brand,
        "impossible_hard": _query_impossible_hard,
    }
    query = builders[category](document)
    return query.strip() if isinstance(query, str) and query.strip() else None


def _add_case(
    records: list[EvalCase],
    *,
    category: str,
    document: SteelProductDocument,
    query: str,
    status: str,
) -> None:
    expected_attributes = _expected_for_query(document, category=category)
    if status == "exact_match":
        hard_eligible = find_hard_eligible_documents(
            [document],
            expected_attributes.brand,
            expected_attributes.dn,
            expected_attributes.pn_bar,
            expected_attributes.connection,
        )
        if not hard_eligible:
            return
        eligible_documents = hard_eligible
    else:
        eligible_documents = []

    eligible_articles = document_articles(eligible_documents)
    preferred_documents = score_preferred_documents(eligible_documents, expected_attributes)
    preferred_articles = document_articles(preferred_documents)
    ld_mapping = build_ld_articles_by_competitor(eligible_documents)
    records.append(
        EvalCase(
            id="",
            category=category,
            query=query,
            expected_status=status,
            expected_attributes=expected_attributes,
            eligible_competitor_articles=eligible_articles,
            preferred_competitor_articles=preferred_articles,
            expected_ld_articles_by_competitor=ld_mapping,
        )
    )


def _select_anchor_documents(
    documents: list[SteelProductDocument],
    *,
    target_count: int = 10,
) -> list[SteelProductDocument]:
    grouped: dict[str, list[SteelProductDocument]] = defaultdict(list)
    for document in documents:
        if not document.brand:
            continue
        grouped[normalize_brand(document.brand) or document.brand].append(document)

    ordered_brands = [
        brand
        for brand in (
            "Temper",
            "ALSO",
            "MARSHAL",
            "Broen",
            "ADL",
            "FORTECA",
            "Бивал",
        )
        if brand in {normalize_brand(item.brand) or item.brand for item in documents if item.brand}
    ]

    def _coverage_score(document: SteelProductDocument) -> tuple[int, str]:
        soft_score = sum(
            1
            for value in (
                document.body_material,
                document.medium,
                document.control,
                document.temperature,
                document.article,
            )
            if value is not None
        )
        if infer_series_from_document(document) is not None:
            soft_score += 1
        return soft_score, document.article

    anchors: list[SteelProductDocument] = []
    for brand in ordered_brands:
        brand_docs = sorted(grouped.get(brand, []), key=_coverage_score, reverse=True)
        if brand_docs:
            anchors.append(brand_docs[0])
    if len(anchors) >= target_count:
        return anchors[:target_count]

    remaining = [
        document
        for document in sorted(documents, key=_coverage_score, reverse=True)
        if document not in anchors
    ]
    for document in remaining:
        anchors.append(document)
        if len(anchors) >= target_count:
            break
    return anchors


def build_v3_cases(
    documents: list[SteelProductDocument],
    *,
    target_count: int = 96,
) -> tuple[list[EvalCase], dict[str, Any]]:
    anchors = _select_anchor_documents(
        documents,
        target_count=min(len(documents), max(8, target_count // 6)),
    )
    records: list[EvalCase] = []
    used_queries: set[str] = set()
    counts: Counter[str] = Counter()

    categories = [
        "hard_only",
        "hard_plus_material",
        "hard_plus_medium",
        "hard_plus_series",
        "hard_plus_article",
        "hard_plus_multiple_soft",
        "missing_dn",
        "missing_pn",
        "missing_connection",
        "russian_alias",
        "compact_syntax",
        "natural_language",
    ]

    for anchor in anchors:
        for category in categories:
            query = _query_for_category(anchor, category)
            if query is None:
                continue
            key = " ".join(query.split()).casefold()
            if key in used_queries:
                continue
            expected = _expected_for_query(anchor, category=category)
            eligible_documents = find_hard_eligible_documents(
                documents,
                expected.brand,
                expected.dn,
                expected.pn_bar,
                expected.connection,
            )
            if not eligible_documents:
                continue
            preferred_documents = score_preferred_documents(eligible_documents, expected)
            records.append(
                EvalCase(
                    id="",
                    category=category,
                    query=query,
                    expected_status="exact_match",
                    expected_attributes=expected,
                    eligible_competitor_articles=document_articles(eligible_documents),
                    preferred_competitor_articles=document_articles(preferred_documents),
                    expected_ld_articles_by_competitor=build_ld_articles_by_competitor(
                        eligible_documents
                    ),
                )
            )
            used_queries.add(key)
            counts[category] += 1

    if anchors:
        primary = anchors[0]
        for category, query_builder, status in (
            ("no_brand", _query_no_brand, "cannot_process"),
            ("impossible_hard", _query_impossible_hard, "not_found"),
        ):
            query = query_builder(primary)
            if not query:
                continue
            key = " ".join(query.split()).casefold()
            if key in used_queries:
                continue
            expected = _expected_for_query(primary, category=category)
            records.append(
                EvalCase(
                    id="",
                    category=category,
                    query=query,
                    expected_status=status,
                    expected_attributes=expected,
                    eligible_competitor_articles=[],
                    preferred_competitor_articles=[],
                    expected_ld_articles_by_competitor={},
                )
            )
            used_queries.add(key)
            counts[category] += 1

    records.sort(
        key=lambda item: (
            CATEGORY_ORDER.index(item.category) if item.category in CATEGORY_ORDER else 999,
            item.query,
        )
    )
    if len(records) > target_count:
        mandatory = [
            record
            for record in records
            if record.category in {"no_brand", "impossible_hard"}
        ]
        optional = [
            record
            for record in records
            if record.category not in {"no_brand", "impossible_hard"}
        ]
        if len(mandatory) >= target_count:
            records = mandatory[:target_count]
        else:
            records = mandatory + optional[: max(0, target_count - len(mandatory))]

    for index, record in enumerate(records, start=1):
        record.id = f"v3_{record.category}_{index:04d}"

    meta = {
        "source_documents": len(documents),
        "anchors": len(anchors),
        "records": len(records),
        "by_category": dict(counts),
        "by_brand": Counter(record.expected_attributes.brand or "none" for record in records),
        "with_dn": sum(1 for record in records if record.expected_attributes.dn is not None),
        "with_pn": sum(1 for record in records if record.expected_attributes.pn_bar is not None),
        "with_connection": sum(
            1 for record in records if record.expected_attributes.connection is not None
        ),
        "with_material": sum(
            1 for record in records if record.expected_attributes.body_material is not None
        ),
        "with_medium": sum(
            1
            for record in records
            if record.expected_attributes.medium is not None
        ),
        "with_series": sum(
            1
            for record in records
            if record.expected_attributes.series is not None
        ),
        "with_article": sum(
            1
            for record in records
            if record.expected_attributes.article is not None
        ),
    }
    return records, meta


def build_generated_v3_cases(
    documents: list[SteelProductDocument],
    *,
    target_count: int = 400,
) -> tuple[list[EvalCase], dict[str, Any]]:
    rng = random.Random(42)
    by_brand: dict[str, list[SteelProductDocument]] = defaultdict(list)
    for document in sorted(
        documents,
        key=lambda item: (normalize_brand(item.brand) or "", item.article),
    ):
        brand = normalize_brand(document.brand) or document.brand or "unknown"
        by_brand[brand].append(document)
    ordered_brands = sorted(by_brand)
    for brand in ordered_brands:
        rng.shuffle(by_brand[brand])

    records: list[EvalCase] = []
    used_queries: set[str] = set()
    exact_categories = (
        "hard_only",
        "hard_plus_material",
        "hard_plus_medium",
        "hard_plus_series",
        "hard_plus_article",
        "hard_plus_multiple_soft",
        "missing_dn",
        "missing_pn",
        "missing_connection",
        "russian_alias",
        "compact_syntax",
        "natural_language",
    )
    negative_categories = (
        ("no_brand", "cannot_process"),
        ("impossible_hard", "not_found"),
    )

    category_weights = {
        "hard_only": 3,
        "hard_plus_material": 2,
        "hard_plus_medium": 2,
        "hard_plus_series": 2,
        "hard_plus_article": 2,
        "hard_plus_multiple_soft": 2,
        "missing_dn": 2,
        "missing_pn": 2,
        "missing_connection": 2,
        "russian_alias": 2,
        "compact_syntax": 2,
        "natural_language": 2,
        "no_brand": 2,
        "impossible_hard": 2,
    }
    total_weight = sum(category_weights.values())
    quotas = {
        category: max(1, (target_count * weight) // total_weight)
        for category, weight in category_weights.items()
    }

    def _maybe_add_exact_case(document: SteelProductDocument, category: str) -> bool:
        if len(records) >= target_count or quotas[category] <= 0:
            return False
        query = _query_for_category(document, category)
        if query is None:
            return False
        key = " ".join(query.split()).casefold()
        if key in used_queries:
            return False
        expected = _expected_for_query(document, category=category)
        eligible_documents = find_hard_eligible_documents(
            documents,
            expected.brand,
            expected.dn,
            expected.pn_bar,
            expected.connection,
        )
        if not eligible_documents:
            return False
        preferred_documents = score_preferred_documents(eligible_documents, expected)
        records.append(
            EvalCase(
                id="",
                category=category,
                query=query,
                expected_status="exact_match",
                expected_attributes=expected,
                eligible_competitor_articles=document_articles(eligible_documents),
                preferred_competitor_articles=document_articles(preferred_documents),
                expected_ld_articles_by_competitor=build_ld_articles_by_competitor(
                    eligible_documents
                ),
            )
        )
        used_queries.add(key)
        quotas[category] -= 1
        return True

    def _maybe_add_negative_case(
        document: SteelProductDocument,
        category: str,
        status: str,
    ) -> bool:
        if len(records) >= target_count or quotas[category] <= 0:
            return False
        query = _query_for_category(document, category)
        if query is None:
            return False
        key = " ".join(query.split()).casefold()
        if key in used_queries:
            return False
        records.append(
            EvalCase(
                id="",
                category=category,
                query=query,
                expected_status=status,
                expected_attributes=_expected_for_query(document, category=category),
                eligible_competitor_articles=[],
                preferred_competitor_articles=[],
                expected_ld_articles_by_competitor={},
            )
        )
        used_queries.add(key)
        quotas[category] -= 1
        return True

    for category in exact_categories:
        for brand in ordered_brands:
            for document in by_brand[brand]:
                _maybe_add_exact_case(document, category)
                if len(records) >= target_count or quotas[category] <= 0:
                    break
            if len(records) >= target_count or quotas[category] <= 0:
                break

    for category, status in negative_categories:
        for brand in ordered_brands:
            for document in by_brand[brand]:
                _maybe_add_negative_case(document, category, status)
                if len(records) >= target_count or quotas[category] <= 0:
                    break
            if len(records) >= target_count or quotas[category] <= 0:
                break

    if len(records) < target_count:
        for category in (*exact_categories, *(item[0] for item in negative_categories)):
            for brand in ordered_brands:
                for document in by_brand[brand]:
                    if category in {"no_brand", "impossible_hard"}:
                        status = "cannot_process" if category == "no_brand" else "not_found"
                        _maybe_add_negative_case(document, category, status)
                    else:
                        _maybe_add_exact_case(document, category)
                    if len(records) >= target_count:
                        break
                if len(records) >= target_count:
                    break
            if len(records) >= target_count:
                break

    records.sort(
        key=lambda item: (
            CATEGORY_ORDER.index(item.category) if item.category in CATEGORY_ORDER else 999,
            item.query,
        )
    )
    if len(records) > target_count:
        records = records[:target_count]

    for index, record in enumerate(records, start=1):
        record.id = f"v3_gen_{index:04d}"

    meta = {
        "source_documents": len(documents),
        "records": len(records),
        "by_category": dict(Counter(record.category for record in records)),
        "by_brand": Counter(record.expected_attributes.brand or "none" for record in records),
        "with_dn": sum(1 for record in records if record.expected_attributes.dn is not None),
        "with_pn": sum(1 for record in records if record.expected_attributes.pn_bar is not None),
        "with_connection": sum(
            1 for record in records if record.expected_attributes.connection is not None
        ),
        "with_material": sum(
            1 for record in records if record.expected_attributes.body_material is not None
        ),
        "with_medium": sum(
            1
            for record in records
            if record.expected_attributes.medium is not None
        ),
        "with_series": sum(
            1
            for record in records
            if record.expected_attributes.series is not None
        ),
        "with_article": sum(
            1
            for record in records
            if record.expected_attributes.article is not None
        ),
    }
    return records, meta


def _write_jsonl(records: Iterable[EvalCase], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return output_path


def build_v3_dataset(
    *,
    csv_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    target_count: int = 96,
) -> tuple[list[EvalCase], dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    documents = build_source_documents_from_frame(frame)
    records, meta = build_v3_cases(documents, target_count=target_count)
    _write_jsonl(records, output_path)
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records, meta


def build_generated_v3_dataset(
    *,
    csv_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path = DEFAULT_GENERATED_DATASET_PATH,
    target_count: int = 400,
) -> tuple[list[EvalCase], dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    documents = build_source_documents_from_frame(frame)
    records, meta = build_generated_v3_cases(documents, target_count=target_count)
    _write_jsonl(records, output_path)
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V3 evaluation datasets.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH, help="Source CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GOLDEN_DATASET_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--generated-output",
        type=Path,
        default=DEFAULT_GENERATED_DATASET_PATH,
        help="Generated dataset output path",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=96,
        help="Number of golden cases to write",
    )
    parser.add_argument(
        "--generated-count",
        type=int,
        default=400,
        help="Number of generated cases to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    golden_records, golden_meta = build_v3_dataset(
        csv_path=args.csv,
        output_path=args.output,
        target_count=args.target_count,
    )
    generated_records, generated_meta = build_generated_v3_dataset(
        csv_path=args.csv,
        output_path=args.generated_output,
        target_count=args.generated_count,
    )
    print(
        json.dumps(
            {
                "golden": {
                    "path": str(args.output),
                    "records": len(golden_records),
                    "meta": golden_meta,
                },
                "generated": {
                    "path": str(args.generated_output),
                    "records": len(generated_records),
                    "meta": generated_meta,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = [
    "build_generated_v3_dataset",
    "build_v3_cases",
    "build_v3_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
