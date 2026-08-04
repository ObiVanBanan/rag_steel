"""CSV profiling utilities for the LD mapping source file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

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
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )


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
