"""Build a standalone article-first V2 evaluation dataset."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from eval.v5_constants import DEFAULT_SOURCE_PATHS
from rag_steel.normalization import normalize_article, normalize_brand, normalize_text
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
MANDATORY_ANCHORS: tuple[str, ...] = (
    "КШ.Ф.П.Р.015.40-01",
    "ТМ.3.03.03.01.200.16.С/С",
)
_PATTERN_PREFIX_RE = re.compile(r"^[^\d]+")


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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _first_text(values: Iterable[Any]) -> str | None:
    for value in values:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if not text or text.casefold() == "nan":
            continue
        return text
    return None


def _first_value(values: Iterable[Any]) -> Any:
    for value in values:
        if _is_missing(value):
            continue
        return value
    return None


def _brand_key(value: Any) -> str | None:
    return normalize_brand(value) or normalize_text(value)


def _pattern_key(article: str) -> str:
    normalized = normalize_article(article)
    article_norm = normalized.article_norm or article.casefold()
    article_compact = normalized.article_compact or article_norm
    prefix_match = _PATTERN_PREFIX_RE.match(article_norm)
    prefix = prefix_match.group(0).strip(".-_/ ") if prefix_match else ""
    if prefix:
        prefix_key = prefix[:10]
    elif article_compact:
        prefix_key = article_compact[:4]
    else:
        prefix_key = "misc"
    head = article_compact[:1] if article_compact else ""
    shape = "digit" if head.isdigit() else "alpha"
    return f"{shape}:{prefix_key.casefold()}"


def _pick_article_alias(
    articles: Iterable[str],
    *,
    preferred_articles: Iterable[str] = (),
) -> str | None:
    unique_articles: list[str] = []
    seen_articles: set[str] = set()
    for article in articles:
        if article in seen_articles:
            continue
        seen_articles.add(article)
        unique_articles.append(article)
    if not unique_articles:
        return None

    preferred_set = set(preferred_articles)

    def score(article: str) -> tuple[int, int, int, str]:
        punctuation_count = sum(1 for char in article if not char.isalnum())
        return (
            1 if article in preferred_set else 0,
            punctuation_count,
            len(article),
            article,
        )

    return max(unique_articles, key=score)


def _identity_from_document(document: SteelProductDocument) -> dict[str, Any] | None:
    article = _first_text([document.article])
    if article is None:
        return None
    brand_key = _brand_key(document.brand)
    if brand_key is None:
        return None
    ld_articles: list[str] = []
    seen_ld: set[str] = set()
    for candidate in document.ld_candidates:
        ld_article = _first_text([candidate.article])
        if ld_article is None or ld_article in seen_ld:
            continue
        seen_ld.add(ld_article)
        ld_articles.append(ld_article)
    if not ld_articles:
        return None
    article_norm = normalize_article(article)
    article_key = article_norm.article_compact or article_norm.article_norm
    if article_key is None:
        return None
    return {
        "article": article,
        "article_key": article_key,
        "brand_key": brand_key,
        "identity_key": f"{article_key}::{brand_key}",
        "family": _product_family(document),
        "source_brand": _first_text([document.brand]),
        "dn": document.dn,
        "pn_bar": document.pn_bar,
        "connection": _first_text([document.connection]),
        "ld_articles": ld_articles,
        "pattern_key": _pattern_key(article),
    }


def _family_from_row_values(name: str | None, body_material: str | None) -> str:
    if name and "затвор" in name.casefold():
        return "butterfly_valve"
    if body_material and "латун" in body_material.casefold():
        return "brass_ball_valve"
    return "ball_valve"


def _source_products_from_frame(
    frame: pd.DataFrame,
    *,
    preferred_articles: Iterable[str] = (),
) -> list[dict[str, Any]]:
    normalized = frame.copy()
    normalized["article"] = normalized["steel_article"].map(lambda value: _first_text([value]))
    normalized["brand_key"] = normalized["steel_brand"].map(_brand_key)
    normalized["article_key"] = normalized["article"].map(
        lambda value: normalize_article(value).article_compact if value is not None else None
    )
    normalized = normalized[
        normalized["article"].notna()
        & normalized["brand_key"].notna()
        & normalized["article_key"].notna()
    ].copy()
    if normalized.empty:
        return []

    brand_counts = normalized.groupby("article_key")["brand_key"].nunique(dropna=True)
    ambiguous_article_keys = {
        article_key for article_key, brand_count in brand_counts.items() if int(brand_count) > 1
    }
    normalized = normalized[~normalized["article_key"].isin(ambiguous_article_keys)].copy()

    products: list[dict[str, Any]] = []
    grouped = normalized.groupby(["article_key", "brand_key"], dropna=False, sort=False)
    for (article_key, brand_key), rows in grouped:
        article_aliases = [
            article
            for article in (_first_text([value]) for value in rows["article"].tolist())
            if article is not None
        ]
        article = _pick_article_alias(article_aliases, preferred_articles=preferred_articles)
        if article is None:
            continue
        ld_articles: list[str] = []
        seen_ld: set[str] = set()
        for value in rows["ld_article"].tolist():
            ld_article = _first_text([value])
            if ld_article is None or ld_article in seen_ld:
                continue
            seen_ld.add(ld_article)
            ld_articles.append(ld_article)
        if not ld_articles:
            continue
        name = _first_text(rows["steel_name"].tolist())
        body_material = _first_text(rows["steel_body_material"].tolist())
        family = _family_from_row_values(name, body_material)
        source_brand = _first_text(rows["steel_brand"].tolist())
        dn = _first_value(rows["steel_dn"].tolist())
        pn_bar = _first_value(rows["steel_pn_bar"].tolist())
        connection = _first_text(rows["steel_connection"].tolist())
        products.append(
            {
                "article": article,
                "article_key": str(article_key),
                "brand_key": str(brand_key),
                "identity_key": f"{article_key}::{brand_key}",
                "family": family,
                "source_brand": source_brand,
                "dn": dn,
                "pn_bar": pn_bar,
                "connection": connection,
                "ld_articles": ld_articles,
                "pattern_key": _pattern_key(article),
                "article_aliases": article_aliases,
            }
        )
    products.sort(
        key=lambda item: (
            str(item["family"]),
            str(item["source_brand"] or ""),
            str(item["article_key"]),
            str(item["article"]),
        )
    )
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


def _candidate_sort_key(product: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(product.get("source_brand") or ""),
        str(product.get("article_key") or ""),
        str(product.get("article") or ""),
        str(product.get("identity_key") or ""),
    )


def _feature_tuples(product: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return (
        ("brand", str(product.get("brand_key") or "")),
        ("dn", str(product.get("dn") if product.get("dn") is not None else "")),
        ("pn", str(product.get("pn_bar") if product.get("pn_bar") is not None else "")),
        ("pattern", str(product.get("pattern_key") or "")),
    )


def _select_diverse_products(
    products: Sequence[dict[str, Any]],
    *,
    count: int,
    mandatory_articles: Iterable[str] = (),
) -> list[dict[str, Any]]:
    if len(products) < count:
        raise ValueError(f"Need {count} products, got {len(products)}")

    sorted_products = sorted(products, key=_candidate_sort_key)
    article_to_product = {str(product["article"]): product for product in sorted_products}
    selected: list[dict[str, Any]] = []
    selected_identity_keys: set[str] = set()
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feature_availability: dict[str, Counter[str]] = defaultdict(Counter)
    for product in sorted_products:
        for dimension, value in _feature_tuples(product):
            feature_availability[dimension][value] += 1

    def register(product: dict[str, Any]) -> None:
        selected.append(product)
        selected_identity_keys.add(str(product["identity_key"]))
        for dimension, value in _feature_tuples(product):
            feature_counts[dimension][value] += 1

    for article in mandatory_articles:
        product = article_to_product.get(article)
        if product is None:
            raise ValueError(
                f"Mandatory anchor {article!r} is not eligible for exact_match selection"
            )
        if str(product["identity_key"]) in selected_identity_keys:
            continue
        register(product)

    remaining = [
        product
        for product in sorted_products
        if str(product["identity_key"]) not in selected_identity_keys
    ]
    while len(selected) < count:
        if not remaining:
            raise ValueError(
                f"Need {count} products, got {len(selected)} after diversity selection"
            )

        best_index = 0
        best_score: tuple[Any, ...] | None = None
        for index, product in enumerate(remaining):
            features = _feature_tuples(product)
            new_dimensions = sum(
                1 for dimension, value in features if feature_counts[dimension][value] == 0
            )
            balance_score = -sum(feature_counts[dimension][value] for dimension, value in features)
            rarity_score = sum(
                1.0 / feature_availability[dimension][value] for dimension, value in features
            )
            score = (
                new_dimensions,
                balance_score,
                rarity_score,
                -index,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        chosen = remaining.pop(best_index)
        register(chosen)

    return selected


def build_article_search_v2_records(
    documents: list[SteelProductDocument],
    *,
    per_family: int = 20,
    mandatory_anchors: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = _eligible_documents(documents)
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for family in FAMILY_ORDER:
        for document in eligible[family]:
            identity = _identity_from_document(document)
            if identity is None:
                continue
            buckets[family].append(identity)
    return _build_records_from_products(
        buckets,
        per_family=per_family,
        mandatory_anchors=tuple(mandatory_anchors),
    )


def _build_records_from_products(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    per_family: int,
    mandatory_anchors: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_family: dict[str, int] = {}
    mandatory_anchor_set = set(mandatory_anchors)

    for family in FAMILY_ORDER:
        family_products = buckets[family]
        family_mandatory = [
            product["article"]
            for product in family_products
            if product["article"] in mandatory_anchor_set
        ]
        selected = _select_diverse_products(
            family_products,
            count=per_family,
            mandatory_articles=family_mandatory,
        )
        by_family[family] = len(selected)
        for ordinal, (product, (variant_name, template)) in enumerate(
            zip(selected, cycle(QUERY_TEMPLATES), strict=False),
            start=1,
        ):
            source_brand = product.get("source_brand")
            connection = product.get("connection")
            records.append(
                _build_record(
                    family=family,
                    ordinal=ordinal,
                    article=str(product["article"]),
                    ld_articles=list(product["ld_articles"]),
                    source_brand=str(source_brand) if source_brand is not None else None,
                    dn=product.get("dn"),
                    pn_bar=product.get("pn_bar"),
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


def _load_source_products(
    csv_paths: Sequence[Path],
    *,
    preferred_articles: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_paths = [path for path in csv_paths if path.exists()]
    if not existing_paths:
        raise ValueError("No source CSVs found for article_search_v2 dataset")
    frame, source_records = load_source_bundle(existing_paths)
    products = _source_products_from_frame(frame, preferred_articles=preferred_articles)
    return products, [record.to_dict() for record in source_records]


def _validate_mandatory_anchors(
    products: Sequence[dict[str, Any]],
    *,
    required_anchors: Iterable[str],
) -> None:
    article_aliases = {
        alias
        for product in products
        for alias in product.get("article_aliases", [str(product["article"])])
    }
    missing = [anchor for anchor in required_anchors if anchor not in article_aliases]
    if missing:
        raise ValueError(
            "Mandatory article_search_v2 anchors are missing from the source bundle: "
            + ", ".join(missing)
        )


def build_article_search_v2_dataset(
    *,
    csv_paths: tuple[Path, ...] = DEFAULT_SOURCE_PATHS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    per_family: int = 20,
    mandatory_anchors: tuple[str, ...] = MANDATORY_ANCHORS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products, source_files = _load_source_products(
        csv_paths,
        preferred_articles=mandatory_anchors,
    )
    _validate_mandatory_anchors(products, required_anchors=mandatory_anchors)

    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for product in products:
        buckets[product["family"]].append(product)
    records, meta = _build_records_from_products(
        buckets,
        per_family=per_family,
        mandatory_anchors=mandatory_anchors,
    )
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
