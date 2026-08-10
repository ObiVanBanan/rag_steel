"""Compare embedding models on the unified LD evaluation dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field
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

from rag_steel.config import MODEL_REGISTRY, QDRANT_URL, get_embedding_model_spec  # noqa: E402
from rag_steel.indexer import build_index  # noqa: E402
from rag_steel.normalization import normalize_article  # noqa: E402
from rag_steel.search_engine import SearchEngine  # noqa: E402

DEFAULT_DATASET_PATH = Path("eval/queries.jsonl")
DEFAULT_SOURCE_CSV = Path("mapping_results.csv")
DEFAULT_OUTPUT_PATH = Path("eval/model_comparison.md")
DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_MODELS = [
    "text-embedding-3-small",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]
TOP_K = 10


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
    embedding_dimension: int
    top_k: int
    model_load_seconds: float
    indexing_time_ms: float
    query_count: int
    evaluated_retrieval_queries: int
    evaluated_no_match_queries: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    cold_query_latency_ms: float | None
    warm_query_latency_ms: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    ram_peak_mb: float | None
    vram_peak_mb: float | None
    index_size_points: int
    no_match_false_positive_rate: float
    per_category_recall_at_k: dict[str, float] = field(default_factory=dict)
    query_examples: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error_type: str | None = None
    error_message: str | None = None


def _failed_result(model_name: str, error: Exception) -> ModelComparisonResult:
    return ModelComparisonResult(
        model_name=model_name,
        collection_name="",
        document_count=0,
        embedding_dimension=0,
        top_k=TOP_K,
        model_load_seconds=0.0,
        indexing_time_ms=0.0,
        query_count=0,
        evaluated_retrieval_queries=0,
        evaluated_no_match_queries=0,
        recall_at_k=0.0,
        precision_at_k=0.0,
        ndcg_at_k=0.0,
        mrr=0.0,
        cold_query_latency_ms=None,
        warm_query_latency_ms=None,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        ram_peak_mb=None,
        vram_peak_mb=None,
        index_size_points=0,
        no_match_false_positive_rate=0.0,
        per_category_recall_at_k={},
        query_examples=[],
        status="failed",
        error_type=error.__class__.__name__,
        error_message=str(error),
    )


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


def _example_from_response(
    *,
    record: EvalQuery,
    response: Any,
    predicted_articles: list[str],
    expected_articles: set[str],
    limit: int,
) -> dict[str, Any]:
    ranked_results: list[dict[str, Any]] = []
    for rank, result in enumerate(getattr(response, "results", [])[:limit], start=1):
        product = getattr(result, "product", None) or {}
        ranked_results.append(
            {
                "rank": rank,
                "article": product.get("article"),
                "article_norm": product.get("article_norm"),
                "name": product.get("name"),
                "score": getattr(result, "score", None),
                "hybrid_score": getattr(result, "hybrid_score", None),
            }
        )

    first_relevant_rank: int | None = None
    for rank, article in enumerate(predicted_articles[:limit], start=1):
        if article in expected_articles:
            first_relevant_rank = rank
            break

    return {
        "query": record.query,
        "category": record.category,
        "expected_ld_articles": sorted(expected_articles),
        "returned_ld_articles": predicted_articles[:limit],
        "hit_count": sum(
            1 for article in predicted_articles[:limit] if article in expected_articles
        ),
        "first_relevant_rank": first_relevant_rank,
        "latency_ms": float(getattr(response, "timing_ms", {}).get("total", 0.0)),
        "results": ranked_results,
    }


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
    load_started = perf_counter()
    shared_model = _model_factory_for(model_name)()
    model_load_seconds = perf_counter() - load_started
    build_started = perf_counter()

    def model_factory() -> Any:
        return shared_model

    spec = get_embedding_model_spec(model_name)
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
    warm_latencies: list[float] = []
    per_category_hits: dict[str, list[float]] = {}
    query_examples: list[dict[str, Any]] = []
    cold_query_latency_ms: float | None = None
    no_match_false_positives = 0
    no_match_count = 0
    evaluated_retrieval_queries = 0

    for index, record in enumerate(dataset):
        response = engine.search(record.query, limit=limit)
        predicted_articles = [_article_from_result(result) for result in response.results]
        predicted_articles = [article for article in predicted_articles if article]
        deduped_articles = list(dict.fromkeys(predicted_articles))
        if len(deduped_articles) != len(predicted_articles):
            raise RuntimeError("Search response contains duplicate LD articles")
        predicted_articles = deduped_articles
        expected_articles = {_normalize_ld_article(item) for item in record.expected_ld_articles}
        query_examples.append(
            _example_from_response(
                record=record,
                response=response,
                predicted_articles=predicted_articles,
                expected_articles=expected_articles,
                limit=limit,
            )
        )
        if expected_articles:
            evaluated_retrieval_queries += 1
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
        else:
            no_match_count += 1
            if predicted_articles:
                no_match_false_positives += 1

        query_latency_ms = float(response.timing_ms.get("total", 0.0))
        if index == 0:
            cold_query_latency_ms = query_latency_ms
        else:
            warm_latencies.append(query_latency_ms)

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
    warm_query_latency_ms = sum(warm_latencies) / len(warm_latencies) if warm_latencies else None
    no_match_false_positive_rate = (
        no_match_false_positives / no_match_count if no_match_count else 0.0
    )

    per_category_recall = {
        category: (sum(values) / len(values) if values else 0.0)
        for category, values in sorted(per_category_hits.items())
        if category != "no_match"
    }

    return ModelComparisonResult(
        model_name=model_name,
        collection_name=build_result.metadata.collection_name,
        document_count=build_result.metadata.document_count,
        embedding_dimension=spec.dimension or build_result.metadata.embedding_dimension,
        top_k=limit,
        model_load_seconds=model_load_seconds,
        indexing_time_ms=indexing_time_ms,
        query_count=len(dataset),
        evaluated_retrieval_queries=evaluated_retrieval_queries,
        evaluated_no_match_queries=no_match_count,
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        precision_at_k=sum(precisions) / len(precisions) if precisions else 0.0,
        ndcg_at_k=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        cold_query_latency_ms=cold_query_latency_ms,
        warm_query_latency_ms=warm_query_latency_ms,
        latency_p50_ms=_percentile(warm_latencies, 50),
        latency_p95_ms=_percentile(warm_latencies, 95),
        ram_peak_mb=ram_peak_mb,
        vram_peak_mb=vram_peak_mb,
        index_size_points=point_count,
        no_match_false_positive_rate=no_match_false_positive_rate,
        per_category_recall_at_k=per_category_recall,
        query_examples=query_examples,
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
    results: list[ModelComparisonResult] = []
    for model_name in models:
        try:
            results.append(
                _evaluate_single_model(
                    model_name=model_name,
                    dataset=dataset,
                    source_csv=source_csv,
                    qdrant_url=qdrant_url,
                    limit=limit,
                    client_factory=client_factory,
                    build_index_fn=build_index_fn,
                )
            )
        except Exception as error:  # noqa: BLE001
            results.append(_failed_result(model_name, error))
    return sorted(
        results,
        key=lambda item: (
            item.status != "completed",
            -item.ndcg_at_k,
            -item.recall_at_k,
            item.latency_p95_ms,
            item.ram_peak_mb if item.ram_peak_mb is not None else float("inf"),
            item.index_size_points,
        ),
    )


def _results_payload(
    results: list[ModelComparisonResult],
    *,
    dataset_path: Path,
    source_csv: Path,
    qdrant_url: str,
    models: list[str],
    run_id: str,
    examples_path: Path | None = None,
) -> dict[str, Any]:
    selected_model = results[0].model_name if results else None
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "source_csv": str(source_csv),
        "qdrant_url": qdrant_url,
        "models": models,
        "selected_model": selected_model,
        "examples_path": str(examples_path) if examples_path is not None else None,
        "results": [asdict(result) for result in results],
    }


def write_results_json(
    results: list[ModelComparisonResult],
    *,
    dataset_path: Path,
    source_csv: Path,
    qdrant_url: str,
    models: list[str],
    output_path: Path,
    run_id: str,
    examples_path: Path | None = None,
) -> dict[str, Any]:
    payload = _results_payload(
        results,
        dataset_path=dataset_path,
        source_csv=source_csv,
        qdrant_url=qdrant_url,
        models=models,
        run_id=run_id,
        examples_path=examples_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_examples_json(
    results: list[ModelComparisonResult],
    *,
    output_path: Path,
    run_id: str,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                "model_name": result.model_name,
                "status": result.status,
                "top_k": result.top_k,
                "query_examples": result.query_examples,
            }
            for result in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


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
    completed_results = [result for result in results if result.status == "completed"]
    lines = [
        "# Model Comparison",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Generated: `{generated_at}`",
        "",
        "## Selection Order",
        "",
        (
            f"Models are ranked by `LD nDCG@{results[0].top_k if results else TOP_K}`, "
            f"then `LD Recall@{results[0].top_k if results else TOP_K}`, "
            "then `p95 latency`, then memory."
        ),
        "",
        (
            f"| Model | nDCG@{results[0].top_k if results else TOP_K} | "
            f"Recall@{results[0].top_k if results else TOP_K} | MRR | "
            f"Precision@{results[0].top_k if results else TOP_K} | "
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
                    f"{result.ndcg_at_k:.4f}",
                    f"{result.recall_at_k:.4f}",
                    f"{result.mrr:.4f}",
                    f"{result.precision_at_k:.4f}",
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

    if completed_results:
        winner = completed_results[0]
        lines.extend(
            [
                "",
                "## Selected Model",
                "",
                f"`{winner.model_name}` ranks first by the plan's tie-break rules.",
                "",
                "Reason:",
                f"- highest `LD nDCG@{winner.top_k}` at {winner.ndcg_at_k:.4f}",
                f"- `LD Recall@{winner.top_k}` at {winner.recall_at_k:.4f}",
                f"- `p95 latency` at {winner.latency_p95_ms:.1f} ms",
                f"- peak RAM at {_format_mb(winner.ram_peak_mb)} MiB",
                "",
                "## Notes",
                "",
                "- `RAM MB` is Python peak traced memory when available.",
                "- `VRAM MB` is reported when CUDA is available.",
                "- `Index points` uses the active Qdrant collection point count.",
                "- `No-match FP rate` is reported only for queries with empty gold sets.",
                "- `query_examples` in the JSON results stores per-query returned "
                "articles and top results.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Selected Model",
                "",
                "No models completed successfully.",
            ]
        )

    failed_results = [result for result in results if result.status != "completed"]
    if failed_results:
        lines.extend(["", "## Failed Models", ""])
        for result in failed_results:
            lines.append(
                f"- `{result.model_name}`: {result.error_type or 'Error'}"
                f"{': ' + result.error_message if result.error_message else ''}"
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
        "--results",
        type=Path,
        default=None,
        help="Machine-readable JSON results path",
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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = args.results or (DEFAULT_RESULTS_DIR / f"{run_id}.json")
    examples_path = results_path.with_name(f"{results_path.stem}_examples.json")
    results = compare_models(
        models=list(args.models),
        dataset_path=args.dataset,
        source_csv=args.csv,
        qdrant_url=args.qdrant_url,
        limit=args.limit,
    )
    write_examples_json(
        results,
        output_path=examples_path,
        run_id=run_id,
    )
    write_results_json(
        results,
        dataset_path=args.dataset,
        source_csv=args.csv,
        qdrant_url=args.qdrant_url,
        models=list(args.models),
        output_path=results_path,
        run_id=run_id,
        examples_path=examples_path,
    )
    report = render_report(results, dataset_path=args.dataset, output_path=args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
