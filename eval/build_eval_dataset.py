"""Build a deterministic LD evaluation dataset from mapping_results.csv."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Make `rag_steel` importable when running this script directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_steel.data_builder import build_source_documents_from_frame  # noqa: E402
from rag_steel.normalization import normalize_article  # noqa: E402

DEFAULT_SOURCE_PATH = Path("mapping_results.csv")
DEFAULT_OUTPUT_PATH = Path("eval/queries.jsonl")
DEFAULT_DOCUMENT_LIMIT = 70


def _format_dn(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _expected_ld_articles(document: Any) -> list[str]:
    articles = {
        candidate.article_norm
        for candidate in getattr(document, "ld_candidates", [])
        if getattr(candidate, "article_norm", "")
    }
    return sorted(articles)


def _modified_article(article: str, ordinal: int) -> str:
    normalized = normalize_article(article)
    compact = normalized.article_compact or re.sub(r"\s+", "", article)
    if not compact:
        return article

    variants = [
        article.lower(),
        article.upper(),
        normalized.article_norm or article.lower(),
        compact,
        re.sub(r"[./_-]+", " ", article).strip(),
        f"{compact[: max(1, len(compact) // 2)]} {compact[max(1, len(compact) // 2):]}",
    ]
    variant = variants[ordinal % len(variants)]
    if variant == article:
        variant = f"{compact[: max(1, len(compact) // 2)]} {compact[max(1, len(compact) // 2):]}"
    return variant


def _partial_article(article: str, ordinal: int) -> str:
    normalized = normalize_article(article)
    compact = normalized.article_compact or re.sub(r"\s+", "", article)
    if not compact:
        return article[: max(1, len(article) // 2)]
    prefix_length = min(len(compact), max(3, len(compact) // 2))
    if ordinal % 2 == 0:
        return compact[:prefix_length]
    return f"{compact[:prefix_length]} {compact[prefix_length:]}".strip()


def _natural_language_query(document: Any) -> str:
    dn = _format_dn(getattr(document, "dn", None))
    pn = _format_dn(getattr(document, "pn_bar", None))
    parts = [
        "Нужен аналог",
        getattr(document, "name", ""),
        f"для DN {dn}" if dn else None,
        f"PN {pn}" if pn else None,
        getattr(document, "connection", None),
        getattr(document, "medium", None),
    ]
    return " ".join(str(part) for part in parts if part)


def _mixed_query(document: Any) -> str:
    dn = _format_dn(getattr(document, "dn", None))
    pn = _format_dn(getattr(document, "pn_bar", None))
    parts = [
        getattr(document, "brand", None),
        getattr(document, "article", None),
        getattr(document, "name", None),
        f"DN{dn}" if dn else None,
        f"PN{pn}" if pn else None,
        getattr(document, "connection", None),
    ]
    return " ".join(str(part) for part in parts if part)


def _build_record(query: str, category: str, expected_ld_articles: list[str]) -> dict[str, Any]:
    return {
        "query": query,
        "category": category,
        "expected_ld_articles": expected_ld_articles,
    }


def build_eval_dataset(
    source_path: Path = DEFAULT_SOURCE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    document_limit: int = DEFAULT_DOCUMENT_LIMIT,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(source_path)
    documents = [
        document
        for document in build_source_documents_from_frame(frame)
        if getattr(document, "ld_candidates", [])
        and getattr(document, "brand", None)
        and getattr(document, "dn", None) is not None
        and getattr(document, "pn_bar", None) is not None
        and getattr(document, "connection", None)
        and getattr(document, "medium", None)
        and getattr(document, "control", None)
    ]

    selected_documents = documents[:document_limit]
    records: list[dict[str, Any]] = []

    for index, document in enumerate(selected_documents):
        expected_ld_articles = _expected_ld_articles(document)
        dn = _format_dn(getattr(document, "dn", None))
        pn = _format_dn(getattr(document, "pn_bar", None))

        records.extend(
            [
                _build_record(document.article, "exact_article", expected_ld_articles),
                _build_record(
                    _modified_article(document.article, index),
                    "modified_article",
                    expected_ld_articles,
                ),
                _build_record(
                    _partial_article(document.article, index),
                    "partial_article",
                    expected_ld_articles,
                ),
                _build_record(document.name, "full_name", expected_ld_articles),
                _build_record(
                    " ".join(
                        part
                        for part in [
                            document.brand,
                            f"DN{dn}" if dn else None,
                            f"PN{pn}" if pn else None,
                        ]
                        if part
                    ),
                    "brand_dn_pn",
                    expected_ld_articles,
                ),
                _build_record(
                    _natural_language_query(document),
                    "natural_language",
                    expected_ld_articles,
                ),
                _build_record(_mixed_query(document), "mixed", expected_ld_articles),
                _build_record(f"no_match_{index:04d}_zxqv", "no_match", []),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the unified LD evaluation dataset.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE_PATH, help="Source CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=DEFAULT_DOCUMENT_LIMIT,
        help="How many source documents to sample",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    records = build_eval_dataset(args.csv, args.output, args.documents)
    print(
        json.dumps(
            {
                "source_csv": str(args.csv),
                "output": str(args.output),
                "documents": args.documents,
                "records": len(records),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
