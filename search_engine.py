"""Compatibility facade for the evolving search pipeline."""

from __future__ import annotations

from rag_steel.search_engine import SearchEngine, SearchResponse, SearchResult

__all__ = [
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
]
