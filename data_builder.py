"""Compatibility wrapper for CSV profiling."""

from src.rag_steel.data_builder import *  # noqa: F401,F403


if __name__ == "__main__":
    from src.rag_steel.data_builder import main

    raise SystemExit(main())
