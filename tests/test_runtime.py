from __future__ import annotations

import pytest

from rag_steel.runtime import SearchBusyError, SearchConcurrencyGate


def test_search_concurrency_gate_rejects_when_busy() -> None:
    gate = SearchConcurrencyGate(1)

    with gate.acquire():
        with pytest.raises(SearchBusyError):
            with gate.acquire():
                pass


def test_search_concurrency_gate_releases_after_exception() -> None:
    gate = SearchConcurrencyGate(1)

    with pytest.raises(RuntimeError, match="boom"):
        with gate.acquire():
            raise RuntimeError("boom")

    with gate.acquire():
        assert True
