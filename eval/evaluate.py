"""Compare embedding models on the unified LD evaluation dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from qdrant_client import QdrantClient

# Make `rag_steel` importable when running this script directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import MODEL_REGISTRY, QDRANT_URL  # noqa: E402
from rag_steel.indexer import build_index  # noqa: E402
from rag_steel.normalization import normalize_article  # noqa: E402
from rag_steel.search_engine import SearchEngine  # noqa: E402

DEFAULT_DATASET_PATH = Path("eval/queries.jsonl")
DEFAULT_SOURCE_CSV = Path("mapping_results.csv")
DEFAULT_OUTPUT_PATH = Path("eval/model_comparison.md")
DEFAULT_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]
TOP_K = 20


@dataclass(slots=True)
class EvalQuery:
    query: str
    category: str
    expected_ld_articles: list[str]


@dataclass(slots=True)
class ModelComparisonResult:
    model_name: str
    collection_name: str
    document_count: int
    indexing_time_ms: float
    query_count: int
    recall_at_20: float
    precision_at_20: float
    ndcg_at_20: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    ram_peak_mb: float | None
    vram_peak_mb: float | None
    index_size_points: int
    per_category_recall_at_20: dict[str, float] = field(default_factory=dict)


def _load_queries(path: Path) -> list[EvalQuery]:
    records: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            EvalQuery(
                query=str(payload["query"]),
                category=str(payload["category"]),
                expected_ld_articles=[str(item) for item in payload["expected_ld_articles"]],
            )
        )
    return records


def _normalize_ld_article(value: Any) -> str:
    normalized = normalize_article(value)
    return normalized.article_norm or str(value).strip().casefold()


def _article_from_result(result: Any) -> str:
    product = getattr(result, "product", None) or {}
    article = product.get("article_norm") or product.get("article")
    return _normalize_ld_article(article) if article else ""


def _dcg(relevances: list[int]) -> float:
    return sum(score / math.log2(index + 2) for index, score in enumerate(relevances))


def _ndcg_at_k(actual_articles: list[str], expected_articles: set[str], k: int) -> float:
    relevances = [1 if article in expected_articles else 0 for article in actual_articles[:k]]
    if not expected_articles:
        return 0.0
    ideal_relevances = [1] * min(len(expected_articles), k)
    ideal = _dcg(ideal_relevances)
    if ideal == 0:
        return 0.0
    return _dcg(relevances) / ideal


def _reciprocal_rank(actual_articles: list[str], expected_articles: set[str]) -> float:
    for index, article in enumerate(actual_articles, start=1):
        if article in expected_articles:
            return 1.0 / index
    return 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def _peak_vram_mb() -> float | None:
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def _model_factory_for(model_name: str) -> Callable[[], Any]:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]


def _default_client_factory(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def _evaluate_single_model(
    *,
    model_name: str,
    dataset: list[EvalQuery],
    source_csv: Path,
    qdrant_url: str,
    limit: int,
    client_factory: Callable[[str], Any],
    build_index_fn: Callable[..., Any],
) -> ModelComparisonResult:
    tracemalloc.start()
    vram_peak_mb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    client = client_factory(qdrant_url)
    build_started = perf_counter()
    model_factory = _model_factory_for(model_name)
    metadata_path = Path("data/reports") / f"index_build_{model_name.replace('/', '_')}.json"
    build_result = build_index_fn(
        source_csv,
        model_name=model_name,
        recreate=False,
        client=client,
        metadata_path=metadata_path,
        model_factory=model_factory,
    )
    indexing_time_ms = (perf_counter() - build_started) * 1000.0

    engine = SearchEngine(
        model_name=model_name,
        qdrant_url=qdrant_url,
        collection_alias=build_result.metadata.collection_name,
        client=client,
        model_factory=model_factory,
    )

    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    per_category_hits: dict[str, list[float]] = {}

    for record in dataset:
        response = engine.search(record.query, limit=limit)
        predicted_articles = [_article_from_result(result) for result in response.results]
        predicted_articles = [article for article in predicted_articles if article]
        expected_articles = {_normalize_ld_article(item) for item in record.expected_ld_articles}
        if expected_articles:
            hit_count = sum(
                1 for article in predicted_articles[:limit] if article in expected_articles
            )
            recalls.append(hit_count / len(expected_articles))
            precisions.append(hit_count / limit)
            ndcgs.append(_ndcg_at_k(predicted_articles, expected_articles, limit))
            reciprocal_ranks.append(_reciprocal_rank(predicted_articles, expected_articles))
            per_category_hits.setdefault(record.category, []).append(
                hit_count / len(expected_articles)
            )
        latencies.append(float(response.timing_ms.get("total", 0.0)))

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    point_count = int(
        client.count(
            collection_name=build_result.metadata.collection_name,
            exact=True,
        ).count
    )
    ram_peak_mb = peak / (1024 * 1024) if peak else None
    vram_peak_mb = _peak_vram_mb()

    per_category_recall = {
        category: (sum(values) / len(values) if values else 0.0)
        for category, values in sorted(per_category_hits.items())
        if category != "no_match"
    }

    return ModelComparisonResult(
        model_name=model_name,
        collection_name=build_result.metadata.collection_name,
        document_count=build_result.metadata.document_count,
        indexing_time_ms=indexing_time_ms,
        query_count=len(dataset),
        recall_at_20=sum(recalls) / len(recalls) if recalls else 0.0,
        precision_at_20=sum(precisions) / len(precisions) if precisions else 0.0,
        ndcg_at_20=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        ram_peak_mb=ram_peak_mb,
        vram_peak_mb=vram_peak_mb,
        index_size_points=point_count,
        per_category_recall_at_20=per_category_recall,
    )


def compare_models(
    *,
    models: list[str],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    qdrant_url: str = QDRANT_URL,
    limit: int = TOP_K,
    client_factory: Callable[[str], Any] = _default_client_factory,
    build_index_fn: Callable[..., Any] = build_index,
) -> list[ModelComparisonResult]:
    dataset = _load_queries(dataset_path)
    results = [
        _evaluate_single_model(
            model_name=model_name,
            dataset=dataset,
            source_csv=source_csv,
            qdrant_url=qdrant_url,
            limit=limit,
            client_factory=client_factory,
            build_index_fn=build_index_fn,
        )
        for model_name in models
    ]
    return sorted(
        results,
        key=lambda item: (
            -item.ndcg_at_20,
            -item.recall_at_20,
            item.latency_p95_ms,
            item.ram_peak_mb if item.ram_peak_mb is not None else float("inf"),
            item.index_size_points,
        ),
    )


def _format_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def render_report(
    results: list[ModelComparisonResult],
    *,
    dataset_path: Path,
    output_path: Path,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Model Comparison",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Generated: `{generated_at}`",
        "",
        "## Selection Order",
        "",
        (
            "Models are ranked by `LD nDCG@20`, then `LD Recall@20`, "
            "then `p95 latency`, then memory."
        ),
        "",
        (
            "| Model | nDCG@20 | Recall@20 | MRR | Precision@20 | "
            "p50 ms | p95 ms | RAM MB | VRAM MB | Index points | "
            "Indexing ms |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.model_name,
                    f"{result.ndcg_at_20:.4f}",
                    f"{result.recall_at_20:.4f}",
                    f"{result.mrr:.4f}",
                    f"{result.precision_at_20:.4f}",
                    f"{result.latency_p50_ms:.1f}",
                    f"{result.latency_p95_ms:.1f}",
                    _format_mb(result.ram_peak_mb),
                    _format_mb(result.vram_peak_mb),
                    str(result.index_size_points),
                    f"{result.indexing_time_ms:.1f}",
                ]
            )
            + " |"
        )

    if results:
        winner = results[0]
        lines.extend(
            [
                "",
                "## Recommended Model",
                "",
                f"`{winner.model_name}` ranks first by the plan's tie-break rules.",
                "",
                "## Notes",
                "",
                "- `RAM MB` is Python peak traced memory when available.",
                "- `VRAM MB` is reported when CUDA is available, otherwise `0.0` or `n/a`.",
                "- `Index points` uses the active Qdrant collection point count.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Recommended Model",
                "",
                "No models were evaluated.",
            ]
        )

    report = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare embedding models on the LD dataset.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Embedding models to compare",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Evaluation dataset JSONL path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help="Source CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TOP_K,
        help="Search result limit",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help="Qdrant URL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    results = compare_models(
        models=list(args.models),
        dataset_path=args.dataset,
        source_csv=args.csv,
        qdrant_url=args.qdrant_url,
        limit=args.limit,
    )
    report = render_report(results, dataset_path=args.dataset, output_path=args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
