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
from eval.v5_constants import CATEGORY_ORDER, DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_SOURCE_PATH
from eval.v5_schema import EvalCase, ExpectedAttributes
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


def _to_v5_expected(expected: V4ExpectedAttributes) -> ExpectedAttributes:
    brand = expected.resolved_brand if expected.raw_brand is not None else None
    resolved_brand = expected.resolved_brand
    resolved_article = expected.resolved_article if expected.resolved_article is not None else None
    if resolved_article is None and expected.article is not None and expected.raw_brand is not None:
        resolved_article = expected.article
    return ExpectedAttributes(
        brand=brand,
        resolved_brand=resolved_brand,
        article=expected.article,
        resolved_article=resolved_article,
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


def _supported_documents(documents: list[SteelProductDocument]) -> list[SteelProductDocument]:
    return [
        document
        for document in documents
        if document.brand and normalize_brand(document.brand) in SUPPORTED_BRANDS
    ]


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


def _transform_v4_records(records: list[V4EvalCase]) -> list[EvalCase]:
    transformed: list[EvalCase] = []
    for record in records:
        expected = record.expected_attributes
        transformed.append(
            EvalCase(
                id=record.id.replace("v4_", "v5_"),
                category=record.category,
                query=record.query,
                expected_status=record.expected_status,
                expected_resolution_mode=record.expected_resolution_mode,
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
            query="Valtec DN50",
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
            category="dn_semantic",
            query="DN57",
            expected_status="cannot_process",
            expected_resolution_mode="no_identity",
            expected_attributes=_build_expected(
                brand=None,
                resolved_brand=None,
                article=None,
                dn=None,
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
    records = _transform_v4_records(base_records)
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
        "negative_cases": negative_cases,
        "brand_cases": brand_cases,
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
    output_path: Path = DEFAULT_GOLDEN_DATASET_PATH,
    target_count: int = 180,
) -> tuple[list[EvalCase], dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    documents = build_source_documents_from_frame(frame)
    records, meta = build_v5_cases(documents, target_count=target_count)
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
