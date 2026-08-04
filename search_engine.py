"""Compatibility facade for the evolving search pipeline."""

from __future__ import annotations

from rag_steel.query_processor import (
    EmbeddingTextAdapter,
    ProcessedQuery,
    QueryProcessor,
    parse_query,
)
from rag_steel.search_engine import SearchEngine, SearchResponse, SearchResult

__all__ = [
    "EmbeddingTextAdapter",
    "ProcessedQuery",
    "QueryProcessor",
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
    "parse_query",
]
