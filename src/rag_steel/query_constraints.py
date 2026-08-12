"""Deterministic query constraint extraction."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from rag_steel.normalization import normalize_brand, normalize_text

_DN_RE = re.compile(r"\b(?:ду|dn)\s*([0-9]{1,4})\b", re.IGNORECASE)
_PN_RE = re.compile(r"\b(?:ру|pn)\s*([0-9]{1,3})\b", re.IGNORECASE)
_SERIES_RE = re.compile(r"\b(?:series|серии?)\s*([0-9]{1,4})\b", re.IGNORECASE)


class QueryConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str | None = None
    dn: int | None = None
    pn_bar: int | None = None
    connection: str | None = None
    series: str | None = None
    body_material: str | None = None


def _extract_brand(query: str) -> str | None:
    brand_aliases = {
        "маршал": "MARSHAL",
        "marshal": "MARSHAL",
        "also": "ALSO",
        "temper": "Temper",
        "adl": "ADL",
        "broen": "Broen",
        "брон": "Broen",
        "forteca": "FORTECA",
    }
    for alias, canonical in brand_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", query, re.IGNORECASE):
            return canonical
    return None


def _extract_body_material(query: str) -> str | None:
    lowered = normalize_text(query) or ""
    if "09г2с" in lowered:
        return "сталь 09г2с"
    if "нержавеющ" in lowered:
        return "нержавеющая сталь"
    match = re.search(r"\b(?:сталь|ст\.?)\s*([0-9]{2,4}[а-яa-z0-9\-]*)\b", lowered)
    if match:
        return f"сталь {match.group(1)}"
    return None


def extract_query_constraints(query: str) -> QueryConstraints:
    normalized = normalize_text(query) or ""
    dn_match = _DN_RE.search(normalized)
    pn_match = _PN_RE.search(normalized)
    series_match = _SERIES_RE.search(normalized)
    connection = None
    if re.search(r"\b(?:фланец|фланцевый|фланцевое|flanged)\b", normalized, re.IGNORECASE):
        connection = "фланцевое"
    elif re.search(r"\b(?:под приварку|приварн(?:ой|ое)|сварн(?:ой|ое)|welded)\b", normalized, re.IGNORECASE):
        connection = "приварное"

    return QueryConstraints(
        brand=_extract_brand(normalized),
        dn=int(dn_match.group(1)) if dn_match else None,
        pn_bar=int(pn_match.group(1)) if pn_match else None,
        connection=connection,
        series=series_match.group(1) if series_match else None,
        body_material=_extract_body_material(normalized),
    )


__all__ = ["QueryConstraints", "extract_query_constraints"]
