"""Build a standalone article-first V2 evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eval.v5_constants import DEFAULT_SOURCE_PATHS
from rag_steel.normalization import normalize_article
from rag_steel.schemas import SteelProductDocument
from rag_steel.source_adapters import load_source_bundle

DEFAULT_OUTPUT_PATH = Path("eval/data/article_search_v2.jsonl")
DEFAULT_META_PATH = Path("eval/data/article_search_v2.meta.json")
QUERY_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("need_analog", "Нужен аналог {article}"),
    ("pick_analog", "Подбери аналог {article}"),
    ("article_only", "{article}"),
    ("replacement_question", "Что есть вместо {article}?"),
)
FAMILY_ORDER: tuple[str, ...] = ("ball_valve", "brass_ball_valve", "butterfly_valve")


def _article_key(document: SteelProductDocument) -> str:
    normalized = normalize_article(document.article)
    return (normalized.article_compact or normalized.article_norm or document.article).casefold()


def _product_family(document: SteelProductDocument) -> str:
    name = (document.name or "").casefold()
    if "затвор" in name:
        return "butterfly_valve"
    if document.body_material and "латун" in document.body_material.casefold():
        return "brass_ball_valve"
    return "ball_valve"


def _first_text(values: Iterable[Any]) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _source_products_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    grouped = frame.groupby("steel_article", dropna=True, sort=False)
    for steel_article, rows in grouped:
        article = _first_text([steel_article])
        if article is None:
            continue
        ld_articles = []
        seen_ld = set()
        for value in rows["ld_article"].tolist():
            text = _first_text([value])
            if text is None or text in seen_ld:
                continue
            seen_ld.add(text)
            ld_articles.append(text)
        if not ld_articles:
            continue
        name = _first_text(rows["steel_name"].tolist())
        body_material = _first_text(rows["steel_body_material"].tolist())
        if name and "затвор" in name.casefold():
            family = "butterfly_valve"
        elif body_material and "латун" in body_material.casefold():
            family = "brass_ball_valve"
        else:
            family = "ball_valve"
        steel_brand_series = rows.get("steel_brand", pd.Series(dtype=object))
        steel_dn = rows["steel_dn"].dropna()
        steel_pn_bar = rows["steel_pn_bar"].dropna()
        products.append(
            {
                "article": article,
                "article_key": (normalize_article(article).article_compact or article.casefold()),
                "family": family,
                "source_brand": _first_text(steel_brand_series.tolist()),
                "dn": steel_dn.iloc[0] if not steel_dn.empty else None,
                "pn_bar": steel_pn_bar.iloc[0] if not steel_pn_bar.empty else None,
                "connection": _first_text(rows["steel_connection"].tolist()),
                "ld_articles": ld_articles,
            }
        )
    products.sort(key=lambda item: (str(item["family"]), str(item["article_key"])))
    return products


def _eligible_documents(
    documents: Iterable[SteelProductDocument],
) -> dict[str, list[SteelProductDocument]]:
    buckets: dict[str, list[SteelProductDocument]] = {family: [] for family in FAMILY_ORDER}
    for document in documents:
        if not document.article or not document.ld_candidates:
            continue
        family = _product_family(document)
        buckets[family].append(document)
    for family in FAMILY_ORDER:
        buckets[family].sort(key=_article_key)
    return buckets


def _build_record(
    *,
    family: str,
    ordinal: int,
    article: str,
    ld_articles: list[str],
    source_brand: str | None,
    dn: Any,
    pn_bar: Any,
    connection: str | None,
    variant_name: str,
    template: str,
) -> dict[str, Any]:
    return {
        "id": f"article_v2_{family}_{ordinal:04d}",
        "category": family,
        "query": template.format(article=article),
        "query_variant": variant_name,
        "article": article,
        "expected_source_article": article,
        "expected_ld_articles": ld_articles,
        "expected_status": "exact_match",
        "source_brand": source_brand,
        "product_family": family,
        "dn": dn,
        "pn_bar": pn_bar,
        "connection": connection,
    }


def build_article_search_v2_records(
    documents: list[SteelProductDocument],
    *,
    per_family: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = _eligible_documents(documents)
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for family in FAMILY_ORDER:
        for document in eligible[family]:
            buckets[family].append(
                {
                    "article": document.article,
                    "family": family,
                    "source_brand": document.brand,
                    "dn": document.dn,
                    "pn_bar": document.pn_bar,
                    "connection": document.connection,
                    "ld_articles": [candidate.article for candidate in document.ld_candidates],
                }
            )
    return _build_records_from_products(buckets, per_family=per_family)


def _build_records_from_products(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_family: dict[str, int] = {}

    for family in FAMILY_ORDER:
        selected = buckets[family][:per_family]
        if len(selected) < per_family:
            raise ValueError(
                f"Not enough eligible {family} documents: need {per_family}, got {len(selected)}"
            )
        by_family[family] = len(selected)
        for ordinal, (document, (variant_name, template)) in enumerate(
            zip(selected, cycle(QUERY_TEMPLATES), strict=False),
            start=1,
        ):
            source_brand = document.get("source_brand")
            connection = document.get("connection")
            records.append(
                _build_record(
                    family=family,
                    ordinal=ordinal,
                    article=str(document["article"]),
                    ld_articles=list(document["ld_articles"]),
                    source_brand=str(source_brand) if source_brand is not None else None,
                    dn=document.get("dn"),
                    pn_bar=document.get("pn_bar"),
                    connection=str(connection) if connection is not None else None,
                    variant_name=variant_name,
                    template=template,
                )
            )

    meta = {
        "records": len(records),
        "per_family": by_family,
        "query_variants": [name for name, _ in QUERY_TEMPLATES],
        "variant_counts": dict(Counter(record["query_variant"] for record in records)),
    }
    return records, meta


def _write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def build_article_search_v2_dataset(
    *,
    csv_paths: tuple[Path, ...] = DEFAULT_SOURCE_PATHS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    per_family: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_paths = [path for path in csv_paths if path.exists()]
    if not existing_paths:
        raise ValueError("No source CSVs found for article_search_v2 dataset")

    if len(existing_paths) == 1:
        frame = pd.read_csv(existing_paths[0])
        source_files: list[dict[str, Any]] = []
    else:
        frame, source_records = load_source_bundle(existing_paths)
        source_files = [record.to_dict() for record in source_records]

    products = _source_products_from_frame(frame)
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for product in products:
        buckets[product["family"]].append(product)
    records, meta = _build_records_from_products(buckets, per_family=per_family)
    meta["source_files"] = source_files
    _write_jsonl(records, output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return records, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build standalone article_search_v2 dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--per-family", type=int, default=20)
    parser.add_argument("--csv", type=Path, nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    csv_paths = tuple(args.csv) if args.csv else DEFAULT_SOURCE_PATHS
    records, meta = build_article_search_v2_dataset(
        csv_paths=csv_paths,
        output_path=args.output,
        meta_path=args.meta,
        per_family=args.per_family,
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "meta_path": str(args.meta),
                "records": len(records),
                "meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
