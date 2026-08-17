"""Build a deterministic V2 evaluation dataset for SearchEngine.search_v2()."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.normalization import normalize_article, normalize_text
from rag_steel.schemas import SteelProductDocument

DEFAULT_SOURCE_PATH = Path("mapping_results.csv")
DEFAULT_OUTPUT_PATH = Path("eval/v2_queries.jsonl")
DEFAULT_DOCUMENT_LIMIT = 40

CONSTRAINT_FIELDS = (
    "brand",
    "dn",
    "pn_bar",
    "connection",
    "series",
    "body_material",
)
PARSER_SUPPORTED_BRANDS = (
    "Temper",
    "Broen",
    "ALSO",
    "MARSHAL",
    "Бивал",
    "ADL",
    "FORTECA",
)
KNOWN_NEGATIVE_MATERIALS = (
    "сталь 20",
    "сталь 09г2с",
    "нержавеющая сталь",
)
UNKNOWN_BRAND_TOKEN = "ZZZTEST"
MOJIBAKE_MARKERS = (
    "\u0420\u00b00486",  # Р°0486
    "\u0420\u201d\u0421\u0453",  # Р”Сѓ
    "\u0421\u201e\u0420\u00bb",  # С„Р»
)


def _normalized_query_key(query: str) -> str:
    return " ".join(query.split()).casefold()


def _is_mojibake_text(value: str | None) -> bool:
    if not value:
        return False
    return any(marker in value for marker in MOJIBAKE_MARKERS)


def _document_has_mojibake(document: SteelProductDocument) -> bool:
    return any(
        _is_mojibake_text(value)
        for value in (
            document.article,
            document.name,
            document.brand,
            document.connection,
            document.body_material,
        )
    )


def _format_number(value: float | None) -> str | None:
    if value is None:
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _normalized_source_article(document: SteelProductDocument) -> str:
    article = normalize_article(document.article).article_norm or document.article_norm
    return article or document.article


def _normalized_ld_articles(document: SteelProductDocument) -> list[str]:
    articles = {
        normalize_article(candidate.article).article_norm or candidate.article_norm
        for candidate in document.ld_candidates
        if normalize_article(candidate.article).article_norm or candidate.article_norm
    }
    return sorted(articles)


def _blank_constraints() -> dict[str, Any]:
    return {field: None for field in CONSTRAINT_FIELDS}


def _constraint_payload(
    *,
    brand: str | None = None,
    dn: int | None = None,
    pn_bar: int | None = None,
    connection: str | None = None,
    series: str | None = None,
    body_material: str | None = None,
) -> dict[str, Any]:
    return {
        "brand": brand,
        "dn": dn,
        "pn_bar": pn_bar,
        "connection": connection,
        "series": series,
        "body_material": body_material,
    }


def _query_id(category: str, ordinal: int) -> str:
    return f"{category}_{ordinal:04d}"


def _make_case(
    *,
    case_id: str,
    query: str,
    category: str,
    gold_mode: str,
    expected_status: str,
    expected_constraints: dict[str, Any] | None = None,
    eligible_competitor_articles: list[str] | None = None,
    expected_ld_articles_by_competitor: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "query": query,
        "category": category,
        "gold_mode": gold_mode,
        "expected_status": expected_status,
        "expected_constraints": expected_constraints or _blank_constraints(),
        "eligible_competitor_articles": eligible_competitor_articles or [],
        "expected_ld_articles_by_competitor": expected_ld_articles_by_competitor or {},
    }


def _find_eligible_documents(
    documents: list[SteelProductDocument],
    expected_constraints: dict[str, Any],
) -> list[SteelProductDocument]:
    eligible: list[SteelProductDocument] = []
    for document in documents:
        if (
            expected_constraints["brand"] is not None
            and document.brand != expected_constraints["brand"]
        ):
            continue
        if expected_constraints["dn"] is not None and document.dn != float(
            expected_constraints["dn"]
        ):
            continue
        if expected_constraints["pn_bar"] is not None and document.pn_bar != float(
            expected_constraints["pn_bar"]
        ):
            continue
        if (
            expected_constraints["connection"] is not None
            and document.connection != expected_constraints["connection"]
        ):
            continue
        if (
            expected_constraints["body_material"] is not None
            and document.body_material != expected_constraints["body_material"]
        ):
            continue
        if expected_constraints["series"] is not None:
            haystack = (
                normalize_text(" ".join([document.name, document.article, document.article_norm]))
                or ""
            )
            if not re.search(rf"\b{re.escape(str(expected_constraints['series']))}\b", haystack):
                continue
        eligible.append(document)
    return eligible


def _gold_from_documents(
    documents: list[SteelProductDocument],
) -> tuple[list[str], dict[str, list[str]]]:
    competitor_articles: list[str] = []
    mapping: dict[str, set[str]] = {}
    for document in documents:
        article = _normalized_source_article(document)
        if article not in mapping:
            competitor_articles.append(article)
            mapping[article] = set()
        mapping[article].update(_normalized_ld_articles(document))
    return competitor_articles, {article: sorted(values) for article, values in mapping.items()}


def _article_gold(
    documents: list[SteelProductDocument],
    article_query: str,
) -> list[SteelProductDocument]:
    article_norm = normalize_article(article_query).article_norm
    if article_norm is None:
        return []
    return [
        document for document in documents if _normalized_source_article(document) == article_norm
    ]


def _name_gold(
    documents: list[SteelProductDocument],
    name_query: str,
) -> list[SteelProductDocument]:
    name_norm = normalize_text(name_query)
    if name_norm is None:
        return []
    return [
        document for document in documents if (normalize_text(document.name) or "") == name_norm
    ]


def _article_modified_variants(article: str) -> list[str]:
    normalized = normalize_article(article)
    compact = normalized.article_compact or article.replace(" ", "")
    variants = [
        article.lower(),
        article.upper(),
        compact,
        re.sub(r"[./_-]+", "", article),
        (
            re.sub(r"(?<=\w)(?=[./_-])", " ", article)
            .replace(".", " ")
            .replace("/", " ")
            .replace("-", " ")
            .replace("_", " ")
        ),
    ]
    deduped: list[str] = []
    for value in variants:
        candidate = " ".join(value.split()).strip()
        if candidate and candidate != article and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _article_partial_variant(article: str) -> str | None:
    compact = normalize_article(article).article_compact
    if not compact or len(compact) < 6:
        return None
    return compact[: max(3, len(compact) // 2)]


def _has_combination(
    documents: list[SteelProductDocument],
    *,
    brand: str | None = None,
    dn: float | None = None,
    pn_bar: float | None = None,
    connection: str | None = None,
    body_material: str | None = None,
) -> bool:
    for document in documents:
        if brand is not None and document.brand != brand:
            continue
        if dn is not None and document.dn != dn:
            continue
        if pn_bar is not None and document.pn_bar != pn_bar:
            continue
        if connection is not None and document.connection != connection:
            continue
        if body_material is not None and document.body_material != body_material:
            continue
        return True
    return False


def _first_missing_pn(
    documents: list[SteelProductDocument],
    anchor: SteelProductDocument,
) -> int | None:
    if anchor.brand is None or anchor.dn is None:
        return None
    for candidate in (999, 998, 997):
        if not _has_combination(
            documents,
            brand=anchor.brand,
            dn=anchor.dn,
            pn_bar=float(candidate),
        ):
            return candidate
    return None


def _first_missing_dn(
    documents: list[SteelProductDocument],
    anchor: SteelProductDocument,
) -> int | None:
    if anchor.brand is None or anchor.pn_bar is None:
        return None
    for candidate in (999, 998, 997):
        if not _has_combination(
            documents,
            brand=anchor.brand,
            dn=float(candidate),
            pn_bar=anchor.pn_bar,
        ):
            return candidate
    return None


def _first_missing_material(
    documents: list[SteelProductDocument],
    anchor: SteelProductDocument,
) -> str | None:
    if anchor.brand is None or anchor.dn is None or anchor.pn_bar is None:
        return None
    for candidate in KNOWN_NEGATIVE_MATERIALS:
        if candidate == anchor.body_material:
            continue
        if not _has_combination(
            documents,
            brand=anchor.brand,
            dn=anchor.dn,
            pn_bar=anchor.pn_bar,
            body_material=candidate,
        ):
            return candidate
    return None


def _first_missing_known_brand(
    documents: list[SteelProductDocument],
    anchor: SteelProductDocument,
) -> str | None:
    if anchor.dn is None or anchor.pn_bar is None:
        return None
    brands = sorted(
        {
            document.brand
            for document in documents
            if (
                document.brand
                and document.brand != anchor.brand
                and document.brand in PARSER_SUPPORTED_BRANDS
            )
        }
    )
    for candidate in brands:
        if not _has_combination(
            documents,
            brand=candidate,
            dn=anchor.dn,
            pn_bar=anchor.pn_bar,
        ):
            return candidate
    return None


def _natural_language_query(document: SteelProductDocument) -> str:
    parts = [
        "Нужен аналог",
        document.brand,
        f"DN{_format_number(document.dn)}" if document.dn is not None else None,
        f"PN{_format_number(document.pn_bar)}" if document.pn_bar is not None else None,
        document.connection,
        document.body_material,
    ]
    return " ".join(str(part) for part in parts if part)


def _natural_language_constraints(document: SteelProductDocument) -> dict[str, Any]:
    return _constraint_payload(
        brand=document.brand,
        dn=int(document.dn) if document.dn is not None else None,
        pn_bar=int(document.pn_bar) if document.pn_bar is not None else None,
        connection=document.connection,
        body_material=document.body_material,
    )


def _parser_only_cases() -> list[dict[str, Any]]:
    cases = [
        ("Temper DN80 PN16", _constraint_payload(brand="Temper", dn=80, pn_bar=16)),
        ("Temper DN80PN16", _constraint_payload(brand="Temper", dn=80, pn_bar=16)),
        ("Temper DN80/PN16", _constraint_payload(brand="Temper", dn=80, pn_bar=16)),
        ("Temper Ду80 Ру16", _constraint_payload(brand="Temper", dn=80, pn_bar=16)),
        ("Temper Ду80Ру16", _constraint_payload(brand="Temper", dn=80, pn_bar=16)),
        ("серия 60", _constraint_payload(series="60")),
        ("серии 60", _constraint_payload(series="60")),
        ("series 60", _constraint_payload(series="60")),
        ("сталь20", _constraint_payload(body_material="сталь 20")),
        ("сталь 20", _constraint_payload(body_material="сталь 20")),
        ("steel 20", _constraint_payload(body_material="сталь 20")),
        ("09Г2С", _constraint_payload(body_material="сталь 09г2с")),
        ("сталь 09Г2С", _constraint_payload(body_material="сталь 09г2с")),
        ("фланцевый", _constraint_payload(connection="фланцевое")),
        ("фланцевое", _constraint_payload(connection="фланцевое")),
        ("сварной", _constraint_payload(connection="сварное")),
        ("приварной", _constraint_payload(connection="сварное")),
        ("резьбовой", _constraint_payload(connection="резьбовое")),
        ("муфтовый", _constraint_payload(connection="муфтовое")),
    ]
    return [
        _make_case(
            case_id=_query_id("parser_only", index + 1),
            query=query,
            category="parser_only",
            gold_mode="parser_only",
            expected_status="parser_only",
            expected_constraints=constraints,
        )
        for index, (query, constraints) in enumerate(cases)
    ]


def build_v2_cases(
    documents: list[SteelProductDocument],
    document_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    usable_documents = [
        document
        for document in documents
        if document.article_norm
        and document.name
        and document.brand
        and document.ld_candidates
        and not _document_has_mojibake(document)
    ]
    skipped_mojibake = sum(1 for document in documents if _document_has_mojibake(document))
    anchors = sorted(usable_documents, key=lambda item: item.article_norm)[:document_limit]

    seen_queries: set[str] = set()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    sequence = 1

    def add_case(record: dict[str, Any]) -> None:
        nonlocal sequence
        key = _normalized_query_key(record["query"])
        if key in seen_queries:
            return
        seen_queries.add(key)
        record["id"] = _query_id(record["category"], sequence)
        sequence += 1
        records.append(record)
        if record["gold_mode"] == "parser_only":
            counts["parser_only"] += 1
        elif record["expected_status"] == "not_found":
            counts["negative"] += 1
        else:
            counts["positive"] += 1

    for anchor in anchors:
        if anchor.brand and anchor.dn is not None and anchor.pn_bar is not None:
            constraints = _constraint_payload(
                brand=anchor.brand,
                dn=int(anchor.dn),
                pn_bar=int(anchor.pn_bar),
            )
            eligible = _find_eligible_documents(usable_documents, constraints)
            competitors, mapping = _gold_from_documents(eligible)
            if competitors:
                add_case(
                    _make_case(
                        case_id="",
                        query=(
                            f"{anchor.brand} DN{_format_number(anchor.dn)} "
                            f"PN{_format_number(anchor.pn_bar)}"
                        ),
                        category="brand_dn_pn",
                        gold_mode="constraints",
                        expected_status="exact_match",
                        expected_constraints=constraints,
                        eligible_competitor_articles=competitors,
                        expected_ld_articles_by_competitor=mapping,
                    )
                )
                add_case(
                    _make_case(
                        case_id="",
                        query=(
                            f"{anchor.brand} DN{_format_number(anchor.dn)}"
                            f"PN{_format_number(anchor.pn_bar)}"
                        ),
                        category="brand_dn_pn_compact",
                        gold_mode="constraints",
                        expected_status="exact_match",
                        expected_constraints=constraints,
                        eligible_competitor_articles=competitors,
                        expected_ld_articles_by_competitor=mapping,
                    )
                )
                add_case(
                    _make_case(
                        case_id="",
                        query=(
                            f"{anchor.brand} Ду{_format_number(anchor.dn)} "
                            f"Ру{_format_number(anchor.pn_bar)}"
                        ),
                        category="brand_du_ru",
                        gold_mode="constraints",
                        expected_status="exact_match",
                        expected_constraints=constraints,
                        eligible_competitor_articles=competitors,
                        expected_ld_articles_by_competitor=mapping,
                    )
                )

                natural_constraints = _natural_language_constraints(anchor)
                natural_eligible = _find_eligible_documents(usable_documents, natural_constraints)
                natural_competitors, natural_mapping = _gold_from_documents(natural_eligible)
                if natural_competitors:
                    add_case(
                        _make_case(
                            case_id="",
                            query=_natural_language_query(anchor),
                            category="natural_language",
                            gold_mode="constraints",
                            expected_status="exact_match",
                            expected_constraints=natural_constraints,
                            eligible_competitor_articles=natural_competitors,
                            expected_ld_articles_by_competitor=natural_mapping,
                        )
                    )

            if anchor.body_material:
                material_constraints = {
                    **constraints,
                    "body_material": anchor.body_material,
                }
                material_eligible = _find_eligible_documents(usable_documents, material_constraints)
                competitors, mapping = _gold_from_documents(material_eligible)
                if competitors:
                    add_case(
                        _make_case(
                            case_id="",
                            query=(
                                f"{anchor.brand} DN{_format_number(anchor.dn)} "
                                f"PN{_format_number(anchor.pn_bar)} {anchor.body_material}"
                            ),
                            category="brand_dn_pn_material",
                            gold_mode="constraints",
                            expected_status="exact_match",
                            expected_constraints=material_constraints,
                            eligible_competitor_articles=competitors,
                            expected_ld_articles_by_competitor=mapping,
                        )
                    )

            if anchor.connection:
                connection_constraints = {
                    **constraints,
                    "connection": anchor.connection,
                }
                connection_eligible = _find_eligible_documents(
                    usable_documents,
                    connection_constraints,
                )
                competitors, mapping = _gold_from_documents(connection_eligible)
                if competitors:
                    add_case(
                        _make_case(
                            case_id="",
                            query=(
                                f"{anchor.brand} DN{_format_number(anchor.dn)} "
                                f"PN{_format_number(anchor.pn_bar)} {anchor.connection}"
                            ),
                            category="brand_dn_pn_connection",
                            gold_mode="constraints",
                            expected_status="exact_match",
                            expected_constraints=connection_constraints,
                            eligible_competitor_articles=competitors,
                            expected_ld_articles_by_competitor=mapping,
                        )
                    )

        article_docs = _article_gold(usable_documents, anchor.article)
        competitors, mapping = _gold_from_documents(article_docs)
        if competitors:
            add_case(
                _make_case(
                    case_id="",
                    query=anchor.article,
                    category="article_exact",
                    gold_mode="article",
                    expected_status="exact_match",
                    eligible_competitor_articles=competitors,
                    expected_ld_articles_by_competitor=mapping,
                )
            )
            modified = _article_modified_variants(anchor.article)
            if modified:
                add_case(
                    _make_case(
                        case_id="",
                        query=modified[0],
                        category="article_modified",
                        gold_mode="article",
                        expected_status="exact_match",
                        eligible_competitor_articles=competitors,
                        expected_ld_articles_by_competitor=mapping,
                    )
                )
            partial = _article_partial_variant(anchor.article)
            if partial:
                add_case(
                    _make_case(
                        case_id="",
                        query=partial,
                        category="article_partial",
                        gold_mode="article",
                        expected_status="exact_match",
                        eligible_competitor_articles=competitors,
                        expected_ld_articles_by_competitor=mapping,
                    )
                )

        name_docs = _name_gold(usable_documents, anchor.name)
        competitors, mapping = _gold_from_documents(name_docs)
        if competitors:
            add_case(
                _make_case(
                    case_id="",
                    query=anchor.name,
                    category="name_exact",
                    gold_mode="name",
                    expected_status="exact_match",
                    eligible_competitor_articles=competitors,
                    expected_ld_articles_by_competitor=mapping,
                )
            )

        missing_pn = _first_missing_pn(usable_documents, anchor)
        if missing_pn is not None and anchor.brand and anchor.dn is not None:
            add_case(
                _make_case(
                    case_id="",
                    query=f"{anchor.brand} DN{_format_number(anchor.dn)} PN{missing_pn}",
                    category="wrong_pn",
                    gold_mode="constraints",
                    expected_status="not_found",
                    expected_constraints=_constraint_payload(
                        brand=anchor.brand,
                        dn=int(anchor.dn),
                        pn_bar=missing_pn,
                    ),
                )
            )

        missing_dn = _first_missing_dn(usable_documents, anchor)
        if missing_dn is not None and anchor.brand and anchor.pn_bar is not None:
            add_case(
                _make_case(
                    case_id="",
                    query=f"{anchor.brand} DN{missing_dn} PN{_format_number(anchor.pn_bar)}",
                    category="wrong_dn",
                    gold_mode="constraints",
                    expected_status="not_found",
                    expected_constraints=_constraint_payload(
                        brand=anchor.brand,
                        dn=missing_dn,
                        pn_bar=int(anchor.pn_bar),
                    ),
                )
            )

        missing_material = _first_missing_material(usable_documents, anchor)
        if (
            missing_material
            and anchor.brand
            and anchor.dn is not None
            and anchor.pn_bar is not None
        ):
            add_case(
                _make_case(
                    case_id="",
                    query=(
                        f"{anchor.brand} DN{_format_number(anchor.dn)} "
                        f"PN{_format_number(anchor.pn_bar)} {missing_material}"
                    ),
                    category="wrong_material",
                    gold_mode="constraints",
                    expected_status="not_found",
                    expected_constraints=_constraint_payload(
                        brand=anchor.brand,
                        dn=int(anchor.dn),
                        pn_bar=int(anchor.pn_bar),
                        body_material=missing_material,
                    ),
                )
            )

        missing_brand = _first_missing_known_brand(usable_documents, anchor)
        if missing_brand and anchor.dn is not None and anchor.pn_bar is not None:
            add_case(
                _make_case(
                    case_id="",
                    query=(
                        f"{missing_brand} DN{_format_number(anchor.dn)} "
                        f"PN{_format_number(anchor.pn_bar)}"
                    ),
                    category="wrong_known_brand",
                    gold_mode="constraints",
                    expected_status="not_found",
                    expected_constraints=_constraint_payload(
                        brand=missing_brand,
                        dn=int(anchor.dn),
                        pn_bar=int(anchor.pn_bar),
                    ),
                )
            )

    if anchors:
        sample_anchor = anchors[0]
        if sample_anchor.dn is not None and sample_anchor.pn_bar is not None:
            add_case(
                _make_case(
                    case_id="",
                    query=(
                        f"{UNKNOWN_BRAND_TOKEN} DN{_format_number(sample_anchor.dn)} "
                        f"PN{_format_number(sample_anchor.pn_bar)}"
                    ),
                    category="unknown_brand",
                    gold_mode="constraints",
                    expected_status="not_found",
                    expected_constraints=_constraint_payload(
                        brand=UNKNOWN_BRAND_TOKEN,
                        dn=int(sample_anchor.dn),
                        pn_bar=int(sample_anchor.pn_bar),
                    ),
                )
            )

    for case in _parser_only_cases():
        add_case(case)

    meta = {
        "source_documents": len(documents),
        "anchors": len(anchors),
        "records": len(records),
        "positive": counts["positive"],
        "negative": counts["negative"],
        "parser_only": counts["parser_only"],
        "skipped_mojibake": skipped_mojibake,
    }
    return records, meta


def build_v2_dataset(
    source_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    document_limit: int = DEFAULT_DOCUMENT_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = pd.read_csv(source_path)
    documents = build_source_documents_from_frame(frame)
    records, meta = build_v2_cases(documents, document_limit=document_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return records, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the V2 search evaluation dataset.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH, help="Source CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=DEFAULT_DOCUMENT_LIMIT,
        help="How many canonical source documents to use as anchors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _, meta = build_v2_dataset(args.csv, args.output, args.documents)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "_find_eligible_documents",
    "_is_mojibake_text",
    "build_v2_cases",
    "build_v2_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
