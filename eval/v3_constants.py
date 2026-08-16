"""Constants for the V3 evaluation suite."""

from __future__ import annotations

from pathlib import Path

HARD_FIELDS: tuple[str, ...] = ("brand", "dn", "pn_bar", "connection")
SOFT_FIELDS: tuple[str, ...] = (
    "body_material",
    "medium",
    "control",
    "temperature",
    "length_mm",
    "series",
    "article",
)
ALL_FIELDS: tuple[str, ...] = HARD_FIELDS + SOFT_FIELDS

DEFAULT_SOURCE_PATH = Path("mapping_results.csv")
DEFAULT_GOLDEN_DATASET_PATH = Path("eval/v3_golden_queries.jsonl")
DEFAULT_GENERATED_DATASET_PATH = Path("eval/v3_generated_queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")

DEFAULT_DEEPSEEK_RESULTS_PATH = DEFAULT_RESULTS_DIR / "deepseek_v3_latest.json"
DEFAULT_RAG_RESULTS_PATH = DEFAULT_RESULTS_DIR / "rag_v3_latest.json"
DEFAULT_E2E_RESULTS_PATH = DEFAULT_RESULTS_DIR / "e2e_v3_latest.json"
DEFAULT_SUMMARY_PATH = Path("eval/v3_summary.md")

CATEGORY_ORDER: tuple[str, ...] = (
    "hard_only",
    "hard_plus_material",
    "hard_plus_medium",
    "hard_plus_series",
    "hard_plus_article",
    "hard_plus_multiple_soft",
    "missing_dn",
    "missing_pn",
    "missing_connection",
    "russian_alias",
    "compact_syntax",
    "natural_language",
    "no_brand",
    "impossible_hard",
)

