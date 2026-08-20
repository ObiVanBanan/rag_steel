"""Constants for the V5 evaluation suite."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SOURCE_PATH = Path("mapping_results.csv")
DEFAULT_GOLDEN_DATASET_PATH = Path("eval/v5_golden_queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")

DEFAULT_DEEPSEEK_RESULTS_PATH = DEFAULT_RESULTS_DIR / "deepseek_v5_latest.json"
DEFAULT_RESOLUTION_RESULTS_PATH = DEFAULT_RESULTS_DIR / "resolution_v5_latest.json"
DEFAULT_RAG_RESULTS_PATH = DEFAULT_RESULTS_DIR / "rag_v5_latest.json"
DEFAULT_E2E_RESULTS_PATH = DEFAULT_RESULTS_DIR / "e2e_v5_latest.json"
DEFAULT_SUMMARY_PATH = Path("eval/v5_summary.md")

CATEGORY_ORDER: tuple[str, ...] = (
    "brand_semantic",
    "brand_typo",
    "mixed_semantic",
    "dn_semantic",
    "pn_semantic",
    "connection_semantic",
    "ambiguous_semantic",
    "unsupported_brand",
    "real_user_regression",
    "article_only_exact",
    "article_only_normalized",
    "article_only_typo",
    "brand_plus_article",
    "article_plus_hard",
    "article_natural_language",
    "unknown_article",
    "ambiguous_article_typo",
    "brand_article_conflict",
    "article_hard_conflict",
    "pn_minimum_semantics",
    "v3_regression",
)
