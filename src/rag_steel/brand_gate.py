"""Cheap competitor brand gate."""

from __future__ import annotations

import re

from rag_steel.competitor_registry import iter_brand_aliases
from rag_steel.normalization import normalize_text

_AMBIGUOUS_ALIASES = {"also"}
_PRODUCT_CONTEXT_PATTERNS = (
    r"\bdn\d*\b",
    r"\bpn\d*\b",
    r"\bdu\d*\b",
    r"\bru\d*\b",
    r"\bкран\b",
    r"\bvalve\b",
    r"\bшаров\w*\b",
    r"\bклапан\w*\b",
    r"\bзадвиж\w*\b",
    r"\bзатвор\w*\b",
)


def _has_product_context(normalized_query: str) -> bool:
    return any(re.search(pattern, normalized_query) for pattern in _PRODUCT_CONTEXT_PATTERNS)


def _alias_matches(normalized_query: str, alias: str) -> bool:
    alias_text = normalize_text(alias) or ""
    if not alias_text:
        return False
    pattern = rf"(?<!\w){re.escape(alias_text)}(?!\w)"
    return re.search(pattern, normalized_query) is not None


def detect_competitor_brand(query: str) -> str | None:
    normalized_query = normalize_text(query) or ""
    if not normalized_query:
        return None

    for alias, canonical in iter_brand_aliases():
        if not _alias_matches(normalized_query, alias):
            continue
        alias_text = normalize_text(alias) or ""
        if alias_text in _AMBIGUOUS_ALIASES and not _has_product_context(normalized_query):
            continue
        return canonical
    return None


__all__ = ["detect_competitor_brand"]
