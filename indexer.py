"""CLI wrapper for the packaged indexer."""

# ruff: noqa: I001

from __future__ import annotations

from rag_steel.indexer import main


if __name__ == "__main__":
    raise SystemExit(main())
