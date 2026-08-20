"""Source file adapters for mapping inputs used by the indexer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from rag_steel.data_builder import REQUIRED_COLUMNS
from rag_steel.normalization import (
    normalize_body_material,
    normalize_brand,
    normalize_connection,
    normalize_supported_brand,
    normalize_text,
)

CANONICAL_COLUMNS = tuple(REQUIRED_COLUMNS) + ("steel_brand",)


@dataclass(slots=True)
class SourceFileRecord:
    name: str
    adapter: str
    sha256: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_text_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_separator(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


def _blank_row() -> dict[str, Any]:
    return {column: None for column in CANONICAL_COLUMNS}


def _canonical_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=CANONICAL_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return frame.loc[:, CANONICAL_COLUMNS]


def _infer_latin_material(text: Any) -> str | None:
    normalized = normalize_text(text)
    if normalized and "латун" in normalized:
        return "латунь"
    return None


def _read_canonical_csv(path: Path) -> tuple[pd.DataFrame, SourceFileRecord]:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    normalized = frame.copy()
    if "steel_brand" not in normalized.columns:
        normalized["steel_brand"] = normalized["steel_name"].map(normalize_brand)
    else:
        normalized["steel_brand"] = normalized["steel_brand"].map(normalize_supported_brand)

    for column in CANONICAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    return (
        normalized.loc[:, CANONICAL_COLUMNS],
        SourceFileRecord(
            name=path.name,
            adapter="canonical",
            sha256=_read_text_sha256(path),
            rows=int(len(normalized)),
        ),
    )


def _read_butterfly_csv(path: Path) -> tuple[pd.DataFrame, SourceFileRecord]:
    frame = pd.read_csv(path, sep=_detect_separator(path))
    required = {
        "ld_marking",
        "ld_dn",
        "ld_pn",
        "ld_connection",
        "ld_control",
        "ld_material",
        "ld_temp",
        "steel_marking",
        "steel_dn",
        "steel_pn",
        "steel_connection",
        "steel_control",
        "steel_material",
        "steel_temp",
        "match_score",
        "match_max",
    }
    missing = [column for column in sorted(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row = _blank_row()
        steel_marking = raw.get("steel_marking")
        row.update(
            {
                "ld_name": raw.get("ld_marking"),
                "ld_article": raw.get("ld_marking"),
                "ld_url": None,
                "ld_dn": raw.get("ld_dn"),
                "ld_pn_mpa": raw.get("ld_pn"),
                "ld_connection": normalize_connection(raw.get("ld_connection")),
                "ld_medium": None,
                "ld_control": raw.get("ld_control"),
                "ld_temp": raw.get("ld_temp"),
                "ld_length": None,
                "steel_name": steel_marking,
                "steel_article": steel_marking,
                "steel_url": None,
                "steel_dn": raw.get("steel_dn"),
                "steel_pn_bar": raw.get("steel_pn"),
                "steel_connection": normalize_connection(raw.get("steel_connection")),
                "steel_body_material": normalize_body_material(raw.get("steel_material")),
                "steel_medium": None,
                "steel_control": raw.get("steel_control"),
                "steel_temp": raw.get("steel_temp"),
                "steel_length": None,
                "match_score": raw.get("match_score"),
                "match_max": raw.get("match_max"),
                "price_ld": None,
                "steel_brand": "PALUR",
            }
        )
        rows.append(row)

    return (
        _canonical_frame(rows),
        SourceFileRecord(
            name=path.name,
            adapter="butterfly",
            sha256=_read_text_sha256(path),
            rows=int(len(frame)),
        ),
    )


def _read_competitor_ld_csv(path: Path) -> tuple[pd.DataFrame, SourceFileRecord]:
    frame = pd.read_csv(path)
    required = {
        "competitor_article",
        "ld_article",
        "c_brand",
        "c_name",
        "c_dn",
        "c_pn",
        "c_thread_type",
        "c_type",
        "name",
        "dn",
        "pn",
        "thread_type",
        "type",
        "price",
        "url",
    }
    missing = [column for column in sorted(required) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row = _blank_row()
        brand = normalize_supported_brand(raw.get("c_brand"))
        material = _infer_latin_material(
            " ".join(str(raw.get(field) or "") for field in ("c_name", "c_type"))
        )
        row.update(
            {
                "ld_name": raw.get("name"),
                "ld_article": raw.get("ld_article"),
                "ld_url": raw.get("url"),
                "ld_dn": raw.get("dn"),
                "ld_pn_mpa": raw.get("pn"),
                "ld_connection": "резьбовое",
                "ld_medium": None,
                "ld_control": None,
                "ld_temp": None,
                "ld_length": None,
                "steel_name": raw.get("c_name"),
                "steel_article": raw.get("competitor_article"),
                "steel_url": None,
                "steel_dn": raw.get("c_dn"),
                "steel_pn_bar": raw.get("c_pn"),
                "steel_connection": "резьбовое",
                "steel_body_material": material,
                "steel_medium": None,
                "steel_control": None,
                "steel_temp": None,
                "steel_length": None,
                "match_score": 1,
                "match_max": 1,
                "price_ld": raw.get("price"),
                "steel_brand": brand,
            }
        )
        rows.append(row)

    return (
        _canonical_frame(rows),
        SourceFileRecord(
            name=path.name,
            adapter="competitor_ld",
            sha256=_read_text_sha256(path),
            rows=int(len(frame)),
        ),
    )


def detect_source_adapter(path: Path) -> str:
    frame = pd.read_csv(path, sep=_detect_separator(path), nrows=0)
    columns = set(frame.columns)
    if set(REQUIRED_COLUMNS).issubset(columns):
        return "canonical"
    if {"ld_marking", "steel_marking"}.issubset(columns):
        return "butterfly"
    if {"competitor_article", "c_brand", "c_name"}.issubset(columns):
        return "competitor_ld"
    raise ValueError(f"Unsupported source schema in {path}")


def load_source_frame(path: Path) -> tuple[pd.DataFrame, SourceFileRecord]:
    adapter = detect_source_adapter(path)
    if adapter == "canonical":
        return _read_canonical_csv(path)
    if adapter == "butterfly":
        return _read_butterfly_csv(path)
    if adapter == "competitor_ld":
        return _read_competitor_ld_csv(path)
    raise ValueError(f"Unsupported adapter for {path}: {adapter}")


def load_source_bundle(paths: Sequence[Path]) -> tuple[pd.DataFrame, list[SourceFileRecord]]:
    frames: list[pd.DataFrame] = []
    source_files: list[SourceFileRecord] = []
    for path in paths:
        frame, record = load_source_frame(path)
        frames.append(frame)
        source_files.append(record)

    if not frames:
        raise ValueError("At least one source CSV is required")

    combined = pd.concat([frame.astype(object) for frame in frames], ignore_index=True)
    combined = combined.loc[:, CANONICAL_COLUMNS]
    return combined, source_files


def combined_source_sha256(source_files: Sequence[SourceFileRecord]) -> str:
    digest = sha256()
    for record in source_files:
        digest.update(record.sha256.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "CANONICAL_COLUMNS",
    "SourceFileRecord",
    "combined_source_sha256",
    "detect_source_adapter",
    "load_source_bundle",
    "load_source_frame",
]
