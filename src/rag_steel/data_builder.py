"""CSV profiling utilities for the LD mapping source file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from rag_steel.normalization import (
    normalize_article,
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
from rag_steel.schemas import LDProduct, SteelProductDocument

REQUIRED_COLUMNS = [
    "ld_name",
    "ld_article",
    "ld_url",
    "ld_dn",
    "ld_pn_mpa",
    "ld_connection",
    "ld_medium",
    "ld_control",
    "ld_temp",
    "ld_length",
    "steel_name",
    "steel_article",
    "steel_url",
    "steel_dn",
    "steel_pn_bar",
    "steel_connection",
    "steel_body_material",
    "steel_medium",
    "steel_control",
    "steel_temp",
    "steel_length",
    "match_score",
    "match_max",
    "price_ld",
]

DEFAULT_REPORT_PATH = Path("data/reports/data_profile.json")
DEFAULT_CONFLICT_LIMIT = 20
DEFAULT_SOURCE_PATH = Path("mapping_results.csv")


@dataclass(slots=True)
class DataProfile:
    source_path: str
    rows: int
    columns: int
    full_duplicates: int
    unique_steel_articles: int
    unique_ld_articles: int
    unique_steel_ld_pairs: int
    null_counts: dict[str, int]
    match_score_distribution: dict[str, int]
    match_max_distribution: dict[str, int]
    conflicting_steel_articles_count: int
    conflicting_steel_articles_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    result: dict[str, int] = {}
    for value, count in counts.items():
        if pd.isna(value):
            key = "null"
        else:
            key = str(value)
        result[key] = int(count)
    return result


def _ensure_required_columns(columns: list[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _conflict_examples(
    df: pd.DataFrame,
    limit: int = DEFAULT_CONFLICT_LIMIT,
) -> list[dict[str, Any]]:
    grouped = df.groupby("steel_article", dropna=False, sort=True)
    examples: list[dict[str, Any]] = []

    for steel_article, group in grouped:
        distinct_ld_articles = group["ld_article"].dropna().nunique()
        if distinct_ld_articles <= 1:
            continue

        example = {
            "steel_article": None if pd.isna(steel_article) else str(steel_article),
            "row_count": int(len(group)),
            "distinct_ld_articles": int(distinct_ld_articles),
            "ld_articles": [
                str(value)
                for value in group["ld_article"].dropna().astype(str).drop_duplicates().head(5)
            ],
            "ld_names": [
                str(value)
                for value in group["ld_name"].dropna().astype(str).drop_duplicates().head(5)
            ],
        }
        examples.append(example)
        if len(examples) >= limit:
            break

    return examples


def profile_csv(csv_path: Path) -> DataProfile:
    """Profile the source CSV and return a structured summary."""

    df = pd.read_csv(csv_path)
    _ensure_required_columns(list(df.columns))

    null_counts = {column: int(df[column].isna().sum()) for column in df.columns}

    return DataProfile(
        source_path=str(csv_path),
        rows=int(len(df)),
        columns=int(len(df.columns)),
        full_duplicates=int(df.duplicated().sum()),
        unique_steel_articles=int(df["steel_article"].nunique(dropna=True)),
        unique_ld_articles=int(df["ld_article"].nunique(dropna=True)),
        unique_steel_ld_pairs=int(df[["steel_article", "ld_article"]].drop_duplicates().shape[0]),
        null_counts=null_counts,
        match_score_distribution=_distribution(df["match_score"]),
        match_max_distribution=_distribution(df["match_max"]),
        conflicting_steel_articles_count=int(
            (df.groupby("steel_article", dropna=False)["ld_article"].nunique(dropna=True) > 1).sum()
        ),
        conflicting_steel_articles_examples=_conflict_examples(df),
    )


def save_profile_report(profile: DataProfile, output_path: Path = DEFAULT_REPORT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _format_id_component(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _select_canonical_value(values: list[Any]) -> Any | None:
    filtered = [value for value in values if not pd.isna(value) and value is not None]
    if not filtered:
        return None

    counts = Counter(filtered)
    top_count = max(counts.values())
    candidates = [value for value, count in counts.items() if count == top_count]
    return sorted(candidates, key=lambda value: str(value))[0]


def _dedupe_text_parts(parts: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part is None:
            continue
        normalized = " ".join(str(part).split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _format_number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:g}"


def _build_semantic_text(
    *,
    name: str,
    brand: str | None,
    dn: float | None,
    pn_bar: float | None,
    connection: str | None,
    body_material: str | None,
    medium: str | None,
    control: str | None,
    temperature: str | None,
    length_mm: float | None,
    article: str | None,
) -> str:
    dn_text = _format_number(dn)
    pn_text = _format_number(pn_bar)
    length_text = _format_number(length_mm)

    parts = _dedupe_text_parts(
        [
            f"Название: {name}" if name else None,
            f"Бренд: {brand}" if brand else None,
            f"Тип товара: {name}" if name else None,
            f"Диаметр DN {dn_text}" if dn_text else None,
            f"Давление PN {pn_text} бар" if pn_text else None,
            f"Соединение: {connection}" if connection else None,
            f"Материал: {body_material}" if body_material else None,
            f"Рабочая среда: {medium}" if medium else None,
            f"Управление: {control}" if control else None,
            f"Температура: {temperature}" if temperature else None,
            f"Длина: {length_text} мм" if length_text else None,
            f"Артикул: {article}" if article else None,
        ]
    )
    return "\n".join(parts)


def _build_lexical_text(
    *,
    name: str,
    name_variants: list[str],
    brand: str | None,
    article: str | None,
    article_norm: str | None,
    article_compact: str | None,
    dn: float | None,
    pn_bar: float | None,
    connection: str | None,
    body_material: str | None,
    medium: str | None,
    control: str | None,
) -> str:
    dn_text = _format_number(dn)
    pn_text = _format_number(pn_bar)

    parts = [
        name or None,
        *name_variants,
        brand,
        article,
        article_norm,
        article_compact,
        f"DN{dn_text}" if dn_text else None,
        f"DN {dn_text}" if dn_text else None,
        f"Ду{dn_text}" if dn_text else None,
        f"Ду {dn_text}" if dn_text else None,
        f"PN{pn_text}" if pn_text else None,
        f"PN {pn_text}" if pn_text else None,
        f"Ру{pn_text}" if pn_text else None,
        f"Ру {pn_text}" if pn_text else None,
        f"{pn_text} бар" if pn_text else None,
        connection,
        body_material,
        medium,
        control,
    ]
    return "\n".join(_dedupe_text_parts(parts))


def _build_ld_product(group: pd.DataFrame) -> LDProduct:
    canonical = group.iloc[0]
    article_norm = normalize_article(canonical["ld_article"]).article_norm
    dn_values = [normalize_dn(value) for value in group["ld_dn"].tolist()]
    pn_values = [normalize_pn_bar(value) for value in group["ld_pn_mpa"].tolist()]
    medium_values = [normalize_medium(value) for value in group["ld_medium"].tolist()]
    control_values = [normalize_control(value) for value in group["ld_control"].tolist()]
    temperature_values = [normalize_temperature(value) for value in group["ld_temp"].tolist()]
    length_values = [normalize_length(value) for value in group["ld_length"].tolist()]
    return LDProduct(
        article=str(_select_canonical_value(group["ld_article"].tolist())),
        article_norm=article_norm or "",
        name=str(_select_canonical_value(group["ld_name"].tolist()) or ""),
        url=_select_canonical_value(group["ld_url"].tolist()),
        dn=_select_canonical_value(dn_values),
        pn_bar=_select_canonical_value(pn_values),
        connection=_select_canonical_value(
            [normalize_connection(value) for value in group["ld_connection"].tolist()]
        ),
        medium=_select_canonical_value(medium_values),
        control=_select_canonical_value(control_values),
        temperature=_select_canonical_value(temperature_values),
        length_mm=_select_canonical_value(length_values),
        price=_select_canonical_value([value for value in group["price_ld"].tolist()]),
    )


def _build_steel_id(
    article_compact: str,
    normalized_name: str,
    dn: float | None,
    pn_bar: float | None,
    connection: str | None,
    control: str | None,
) -> str:
    token = (
        _format_id_component(article_compact)
        + _format_id_component(normalized_name)
        + _format_id_component(dn)
        + _format_id_component(pn_bar)
        + _format_id_component(connection)
        + _format_id_component(control)
    )
    return sha1(token.encode("utf-8")).hexdigest()


def _normalize_source_row(row: pd.Series) -> dict[str, Any]:
    article = normalize_article(row["steel_article"])
    name = normalize_text(row["steel_name"])
    return {
        "article": str(row["steel_article"]),
        "article_norm": article.article_norm or "",
        "article_compact": article.article_compact or "",
        "name": str(_select_canonical_value([row["steel_name"]]) or ""),
        "name_norm": name or "",
        "brand": normalize_brand(row["steel_name"]),
        "dn": normalize_dn(row["steel_dn"]),
        "pn_bar": normalize_pn_bar(row["steel_pn_bar"]),
        "connection": normalize_connection(row["steel_connection"]),
        "body_material": normalize_text(row["steel_body_material"]),
        "medium": normalize_medium(row["steel_medium"]),
        "control": normalize_control(row["steel_control"]),
        "temperature": normalize_temperature(row["steel_temp"]),
        "length_mm": normalize_length(row["steel_length"]),
        "url": _select_canonical_value([row["steel_url"]]),
    }


def _build_source_document(group_rows: pd.DataFrame) -> SteelProductDocument:
    source_rows = group_rows.copy()
    canonical_name = _select_canonical_value(source_rows["steel_name"].tolist()) or ""
    canonical_url = _select_canonical_value(source_rows["steel_url"].tolist())
    canonical_brand = normalize_brand(canonical_name)
    canonical_dn = _select_canonical_value(
        [normalize_dn(value) for value in source_rows["steel_dn"].tolist()]
    )
    canonical_pn = _select_canonical_value(
        [normalize_pn_bar(value) for value in source_rows["steel_pn_bar"].tolist()]
    )
    canonical_connection = _select_canonical_value(
        [normalize_connection(value) for value in source_rows["steel_connection"].tolist()]
    )
    canonical_body_material = _select_canonical_value(
        [normalize_text(value) for value in source_rows["steel_body_material"].tolist()]
    )
    canonical_medium = _select_canonical_value(
        [normalize_medium(value) for value in source_rows["steel_medium"].tolist()]
    )
    canonical_control = _select_canonical_value(
        [normalize_control(value) for value in source_rows["steel_control"].tolist()]
    )
    canonical_temperature = _select_canonical_value(
        [normalize_temperature(value) for value in source_rows["steel_temp"].tolist()]
    )
    canonical_length = _select_canonical_value(
        [normalize_length(value) for value in source_rows["steel_length"].tolist()]
    )
    article = normalize_article(source_rows.iloc[0]["steel_article"])
    normalized_name = normalize_text(canonical_name) or ""

    ld_candidates: list[LDProduct] = []
    ld_article_norms = source_rows["ld_article"].map(
        lambda value: normalize_article(value).article_norm or ""
    )
    for _, ld_group in source_rows.groupby(ld_article_norms, sort=True):
        candidate = _build_ld_product(ld_group)
        if candidate.article_norm:
            ld_candidates.append(candidate)

    steel_id = _build_steel_id(
        article.article_compact or "",
        normalized_name,
        canonical_dn,
        canonical_pn,
        canonical_connection,
        canonical_control,
    )

    name_variants = sorted(
        {
            str(value)
            for value in source_rows["steel_name"].tolist()
            if not pd.isna(value) and value
        },
        key=str,
    )
    semantic_text = _build_semantic_text(
        name=canonical_name,
        brand=canonical_brand,
        dn=canonical_dn,
        pn_bar=canonical_pn,
        connection=canonical_connection,
        body_material=canonical_body_material,
        medium=canonical_medium,
        control=canonical_control,
        temperature=canonical_temperature,
        length_mm=canonical_length,
        article=article.article_raw or article.article_norm,
    )
    lexical_text = _build_lexical_text(
        name=canonical_name,
        name_variants=name_variants or [canonical_name],
        brand=canonical_brand,
        article=article.article_raw,
        article_norm=article.article_norm,
        article_compact=article.article_compact,
        dn=canonical_dn,
        pn_bar=canonical_pn,
        connection=canonical_connection,
        body_material=canonical_body_material,
        medium=canonical_medium,
        control=canonical_control,
    )

    return SteelProductDocument(
        steel_id=steel_id,
        article=str(_select_canonical_value(source_rows["steel_article"].tolist()) or ""),
        article_norm=article.article_norm or "",
        article_compact=article.article_compact or "",
        name=canonical_name,
        name_variants=name_variants or [canonical_name],
        brand=canonical_brand,
        dn=canonical_dn,
        pn_bar=canonical_pn,
        connection=canonical_connection,
        body_material=canonical_body_material,
        medium=canonical_medium,
        control=canonical_control,
        temperature=canonical_temperature,
        length_mm=canonical_length,
        url=canonical_url,
        semantic_text=semantic_text,
        lexical_text=lexical_text,
        ld_candidates=sorted(ld_candidates, key=lambda item: item.article_norm),
    )


def build_source_documents_from_frame(df: pd.DataFrame) -> list[SteelProductDocument]:
    """Group mapping rows into stable source-product documents."""

    rows = df.drop_duplicates().reset_index(drop=True)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for _, row in rows.iterrows():
        source = _normalize_source_row(row)
        key = (
            source["article_compact"],
            source["name_norm"],
            source["dn"],
            source["pn_bar"],
            source["connection"],
            source["control"],
        )

        grouped.setdefault(key, []).append(source | {"row": row})

    documents: list[SteelProductDocument] = []
    for bucket_rows in grouped.values():
        bucket_df = pd.DataFrame([item["row"] for item in bucket_rows])
        documents.append(_build_source_document(bucket_df))

    return sorted(documents, key=lambda item: item.steel_id)


def build_source_documents(csv_path: Path = DEFAULT_SOURCE_PATH) -> list[SteelProductDocument]:
    return build_source_documents_from_frame(pd.read_csv(csv_path))


def build_profile_report(csv_path: Path, output_path: Path = DEFAULT_REPORT_PATH) -> DataProfile:
    profile = profile_csv(csv_path)
    save_profile_report(profile, output_path)
    return profile


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the LD mapping CSV file.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to mapping_results.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON profile report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        profile = build_profile_report(args.csv, args.output)
    except ValueError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(
        json.dumps(
            {
                "rows": profile.rows,
                "columns": profile.columns,
                "full_duplicates": profile.full_duplicates,
                "unique_steel_articles": profile.unique_steel_articles,
                "unique_ld_articles": profile.unique_ld_articles,
                "unique_steel_ld_pairs": profile.unique_steel_ld_pairs,
                "report": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
