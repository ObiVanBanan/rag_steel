"""Shared user-facing messages for search failures."""

from __future__ import annotations

from rag_steel.competitor_registry import COMPETITOR_BRANDS


def search_failure_message() -> str:
    brands = ", ".join(COMPETITOR_BRANDS)
    return f"Подходящие товары не найдены. Возможен поиск по следующим брендам: {brands}"


SEARCH_FAILURE_MESSAGE = search_failure_message()


__all__ = ["SEARCH_FAILURE_MESSAGE", "search_failure_message"]
