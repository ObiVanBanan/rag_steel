"""Build and publish a versioned Qdrant index for steel products."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
from qdrant_client import QdrantClient, models

from config import (
    DEFAULT_MODEL_NAME,
    DENSE_BATCH_SIZE,
    MODEL_REGISTRY,
    QDRANT_COLLECTION_ALIAS,
    QDRANT_URL,
    get_embedding_model_spec,
)
from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.query_processor import EmbeddingTextAdapter
from rag_steel.schemas import SteelProductDocument

DEFAULT_INDEX_METADATA_PATH = Path("data/reports/index_build.json")
DEFAULT_SMOKE_QUERIES = [
    "1184399",
    "а0486",
    "Temper DN80 PN16",
    "Broen Ду80 Ру16",
    "фланцевый кран Ду50 Ру40",
]


@dataclass(slots=True)
class IndexBuildMetadata:
    csv_sha256: str
    embedding_model: str
    embedding_dimension: int
    build_timestamp: str
    document_count: int
    source_row_count: int
    deduplicated_row_count: int
    collection_name: str
    collection_alias: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexBuildResult:
    metadata: IndexBuildMetadata
    metadata_path: Path
    smoke_queries: list[dict[str, Any]]


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return slug or "model"


def _timestamp_string(moment: datetime | None = None) -> str:
    value = moment or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_collection_name(client: QdrantClient, base_name: str) -> str:
    if not client.collection_exists(base_name):
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if not client.collection_exists(candidate):
            return candidate
        suffix += 1


def _encode_dense_batch(
    model: Any,
    documents: list[SteelProductDocument],
    batch_size: int,
    *,
    model_name: str,
) -> list[list[float]]:
    adapter = EmbeddingTextAdapter(model_name=model_name)
    texts = [adapter.prepare_document(document.semantic_text) for document in documents]
    spec = get_embedding_model_spec(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=spec.normalize_embeddings,
        show_progress_bar=True,
    )
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [list(vector) for vector in vectors]


def _build_points(
    documents: list[SteelProductDocument],
    dense_vectors: Sequence[Sequence[float]],
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for document, dense_vector in zip(documents, dense_vectors, strict=True):
        payload = document.model_dump(mode="json")
        points.append(
            models.PointStruct(
                id=document.steel_id,
                vector={
                    "dense": list(dense_vector),
                    "sparse": models.Document(text=document.lexical_text, model="qdrant/bm25"),
                },
                payload=payload,
            )
        )
    return points


def _batch_iter(
    items: list[SteelProductDocument],
    batch_size: int,
) -> list[list[SteelProductDocument]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _create_versioned_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
    metadata: IndexBuildMetadata,
) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata=metadata.to_dict(),
    )


def _upsert_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list[SteelProductDocument],
    model: Any,
    batch_size: int,
    *,
    model_name: str,
) -> None:
    for batch in _batch_iter(documents, batch_size):
        dense_vectors = _encode_dense_batch(
            model,
            batch,
            batch_size,
            model_name=model_name,
        )
        points = _build_points(batch, dense_vectors)
        client.upsert(collection_name=collection_name, points=points, wait=True)


def _extract_query_points(response: Any) -> list[Any]:
    points = getattr(response, "points", None)
    if points is None:
        points = getattr(response, "result", None)
    if points is None:
        return []
    return list(points)


def _run_smoke_queries(
    client: QdrantClient,
    collection_name: str,
    model: Any,
    queries: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    query_vectors = model.encode(
        list(queries),
        batch_size=len(queries),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if hasattr(query_vectors, "tolist"):
        query_vectors = query_vectors.tolist()

    results: list[dict[str, Any]] = []
    for query_text, dense_vector in zip(queries, query_vectors, strict=True):
        response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=list(dense_vector), using="dense", limit=limit),
                models.Prefetch(
                    query=models.Document(text=query_text, model="qdrant/bm25"),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        hits = _extract_query_points(response)
        if not hits:
            raise RuntimeError(f"Smoke query returned no hits: {query_text}")
        results.append(
            {
                "query": query_text,
                "hit_count": len(hits),
                "top_id": getattr(hits[0], "id", None),
            }
        )

    return results


def _sample_payload(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        raise RuntimeError("Collection was created but contains no points")

    payload = records[0].payload or {}
    if not payload:
        raise RuntimeError("Sample payload is empty")
    return payload


def _switch_alias(
    client: QdrantClient,
    collection_name: str,
    alias_name: str,
) -> None:
    aliases = client.get_aliases()
    existing = [
        alias
        for alias in getattr(aliases, "aliases", [])
        if getattr(alias, "alias_name", None) == alias_name
    ]
    if existing and all(
        getattr(alias, "collection_name", None) == collection_name for alias in existing
    ):
        return

    operations: list[Any] = []
    if existing:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias_name)
        )
    )
    client.update_collection_aliases(operations)


def build_index(
    csv_path: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    recreate: bool = False,
    client: QdrantClient | None = None,
    metadata_path: Path = DEFAULT_INDEX_METADATA_PATH,
    batch_size: int = DENSE_BATCH_SIZE,
    build_time: datetime | None = None,
    smoke_queries: Sequence[str] = DEFAULT_SMOKE_QUERIES,
    model_factory: Callable[[], Any] | None = None,
) -> IndexBuildResult:
    df = pd.read_csv(csv_path)
    documents = build_source_documents_from_frame(df)

    factory = model_factory or MODEL_REGISTRY[model_name]
    model = factory()
    embedding_dimension = int(model.get_sentence_embedding_dimension())
    spec = get_embedding_model_spec(model_name)
    if spec.embedding_dimension and embedding_dimension != spec.embedding_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch for "
            f"{model_name}: expected {spec.embedding_dimension}, got {embedding_dimension}"
        )
    timestamp = _timestamp_string(build_time)
    base_collection_name = f"steel_products_{_slugify_model_name(model_name)}_{timestamp}"

    qdrant_client = client or QdrantClient(url=QDRANT_URL)
    collection_name = _unique_collection_name(qdrant_client, base_collection_name)

    metadata = IndexBuildMetadata(
        csv_sha256=_sha256_file(csv_path),
        embedding_model=model_name,
        embedding_dimension=embedding_dimension,
        build_timestamp=timestamp,
        document_count=len(documents),
        source_row_count=len(df),
        deduplicated_row_count=len(df.drop_duplicates()),
        collection_name=collection_name,
        collection_alias=QDRANT_COLLECTION_ALIAS,
    )

    _create_versioned_collection(qdrant_client, collection_name, embedding_dimension, metadata)
    _upsert_documents(
        qdrant_client,
        collection_name,
        documents,
        model,
        batch_size,
        model_name=model_name,
    )

    point_count = qdrant_client.count(collection_name=collection_name, exact=True).count
    if point_count != len(documents):
        raise RuntimeError(
            f"Collection point count mismatch: expected {len(documents)}, got {point_count}"
        )

    _sample_payload(qdrant_client, collection_name)
    smoke_results = _run_smoke_queries(
        qdrant_client,
        collection_name,
        model,
        smoke_queries,
        limit=min(5, len(documents) or 1),
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if recreate:
        _switch_alias(qdrant_client, collection_name, QDRANT_COLLECTION_ALIAS)

    return IndexBuildResult(
        metadata=metadata,
        metadata_path=metadata_path,
        smoke_queries=smoke_results,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a versioned Qdrant index.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to mapping_results.csv")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Switch the active alias to the new collection after smoke checks",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_INDEX_METADATA_PATH,
        help="Where to write the build metadata JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    result = build_index(
        args.csv,
        model_name=args.model,
        recreate=args.recreate,
        metadata_path=args.metadata_path,
    )
    print(
        json.dumps(
            {
                "collection_name": result.metadata.collection_name,
                "collection_alias": result.metadata.collection_alias,
                "document_count": result.metadata.document_count,
                "source_row_count": result.metadata.source_row_count,
                "deduplicated_row_count": result.metadata.deduplicated_row_count,
                "metadata_path": str(result.metadata_path),
                "smoke_queries": result.smoke_queries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
