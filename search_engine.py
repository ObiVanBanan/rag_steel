"""Compatibility facade for the evolving search pipeline."""

from __future__ import annotations

from typing import Any

from rag_steel.query_processor import (
    EmbeddingTextAdapter,
    ProcessedQuery,
    QueryProcessor,
    parse_query,
)


class SearchEngine:
    """Temporary compatibility shim for pre-phase-8 imports."""

    def __init__(self, model_name: str | None = None, **_: Any) -> None:
        self.model_name = model_name
        self.query_processor = QueryProcessor(model_name=model_name or "")

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError("SearchEngine.search is implemented in phase 8")

    def find_analogs(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError("find_analogs is implemented in phase 8")

    def compare_models(
        self,
        query: str,
        models: list[str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        raise NotImplementedError("compare_models is implemented in phase 13")


__all__ = [
    "EmbeddingTextAdapter",
    "ProcessedQuery",
    "QueryProcessor",
    "SearchEngine",
    "parse_query",
]
