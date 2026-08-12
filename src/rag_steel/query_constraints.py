"""Deterministic query constraint extraction."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from rag_steel.normalization import normalize_connection, normalize_text

_DN_PN_RE = re.compile(
    r"(?:dn|\u0434\u0443)\s*([0-9]{1,4})\s*(?:[/\\-]?\s*)?(?:pn|\u0440\u0443)\s*([0-9]{1,3})",
    re.IGNORECASE,
)
_DN_RE = re.compile(r"(?:dn|\u0434\u0443)\s*([0-9]{1,4})", re.IGNORECASE)
_PN_RE = re.compile(r"(?:pn|\u0440\u0443)\s*([0-9]{1,3})", re.IGNORECASE)
_SERIES_RE = re.compile(r"(?:series|\u0441\u0435\u0440\u0438\u044f|\u0441\u0435\u0440\u0438\u0438)\s*([0-9]{1,4})", re.IGNORECASE)
_BRAND_ALIASES = {
    "temper": "Temper",
    "broen": "Broen",
    "also": "ALSO",
    "\u0430\u043b\u0441\u043e": "ALSO",
    "marshal": "MARSHAL",
    "\u043c\u0430\u0440\u0448\u0430\u043b": "MARSHAL",
    "bival": "\u0411\u0438\u0432\u0430\u043b",
    "\u0431\u0438\u0432\u0430\u043b": "\u0411\u0438\u0432\u0430\u043b",
    "adl": "ADL",
    "\u0431\u0440\u043e\u043d": "Broen",
    "forteca": "FORTECA",
}
_CONNECTION_PATTERNS = (
    re.compile(r"\b(?:flanged|\u0444\u043b\u0430\u043d\u0446\u0435\u0432\u043e\u0435|\u0444\u043b\u0430\u043d\u0446\u0435\u0432\u044b\u0439|\u0444\u043b\u0430\u043d\u0435\u0446)\b", re.IGNORECASE),
    re.compile(r"\b(?:welded|\u0441\u0432\u0430\u0440\u043d\u043e\u0439|\u0441\u0432\u0430\u0440\u043d\u043e\u0435|\u043f\u0440\u0438\u0432\u0430\u0440\u043d\u043e\u0439|\u043f\u0440\u0438\u0432\u0430\u0440\u043d\u043e\u0435|\u043f\u043e\u0434 \u043f\u0440\u0438\u0432\u0430\u0440\u043a\u0443)\b", re.IGNORECASE),
    re.compile(r"\b(?:threaded|\u0440\u0435\u0437\u044c\u0431\u043e\u0432\u043e\u0435|\u0440\u0435\u0437\u044c\u0431\u043e\u0432\u043e\u0439|\u0440\u0435\u0437\u044c\u0431\u0430)\b", re.IGNORECASE),
    re.compile(r"\b(?:coupling|\u043c\u0443\u0444\u0442\u043e\u0432\u043e\u0435|\u043c\u0443\u0444\u0442\u043e\u0432\u044b\u0439)\b", re.IGNORECASE),
)
_BODY_MATERIAL_PATTERNS = (
    re.compile(r"\b(?:steel|\u0441\u0442\u0430\u043b\u044c)\s*([0-9]{2,4}[\u0430-\u044fa-z0-9\-]*)\b", re.IGNORECASE),
    re.compile(r"\b09\u04332\u0441\b", re.IGNORECASE),
    re.compile(r"\b\u043d\u0435\u0440\u0436\u0430\u0432\u0435\u044e\u0449\w*\s*\u0441\u0442\u0430\u043b\u044c\b", re.IGNORECASE),
)


class QueryConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str | None = None
    dn: int | None = None
    pn_bar: int | None = None
    connection: str | None = None
    series: str | None = None
    body_material: str | None = None


def _extract_brand(query: str) -> str | None:
    for alias, canonical in _BRAND_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", query, re.IGNORECASE):
            return canonical
    return None


def _extract_connection(query: str) -> str | None:
    for pattern in _CONNECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            return normalize_connection(match.group(0))
    return None


def _extract_body_material(query: str) -> str | None:
    lowered = normalize_text(query) or ""
    for pattern in _BODY_MATERIAL_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        if match.lastindex:
            return normalize_text(f"сталь {match.group(1)}")
        if "09г2с" in match.group(0):
            return normalize_text("сталь 09г2с")
        return normalize_text("нержавеющая сталь")
    return None


def extract_query_constraints(query: str) -> QueryConstraints:
    normalized = normalize_text(query) or ""

    dn = None
    pn = None
    concatenated_match = _DN_PN_RE.search(normalized)
    if concatenated_match:
        dn = int(concatenated_match.group(1))
        pn = int(concatenated_match.group(2))
    else:
        dn_match = _DN_RE.search(normalized)
        pn_match = _PN_RE.search(normalized)
        if dn_match:
            dn = int(dn_match.group(1))
        if pn_match:
            pn = int(pn_match.group(1))

    series_match = _SERIES_RE.search(normalized)

    return QueryConstraints(
        brand=_extract_brand(normalized),
        dn=dn,
        pn_bar=pn,
        connection=_extract_connection(normalized),
        series=series_match.group(1) if series_match else None,
        body_material=_extract_body_material(normalized),
    )


__all__ = ["QueryConstraints", "extract_query_constraints"]
