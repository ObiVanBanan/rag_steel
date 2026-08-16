"""Cheap competitor brand gate."""

from __future__ import annotations

from rag_steel.competitor_registry import iter_brand_aliases
from rag_steel.normalization import normalize_text


def detect_competitor_brand(query: str) -> str | None:
    normalized_query = normalize_text(query) or ""
    if not normalized_query:
        return None

    for alias, canonical in iter_brand_aliases():
        alias_text = normalize_text(alias) or ""
        if not alias_text:
            continue
        if alias_text in normalized_query:
            return canonical
    return None


__all__ = ["detect_competitor_brand"]
