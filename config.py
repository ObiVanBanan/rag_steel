"""Configuration for embedding and Qdrant indexing."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _make_sentence_transformer_factory(model_name: str) -> Callable[[], Any]:
    def factory() -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    return factory


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    model_name: str
    embedding_dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    normalize_embeddings: bool = True


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    "all-MiniLM-L6-v2": _make_sentence_transformer_factory("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": _make_sentence_transformer_factory("all-mpnet-base-v2"),
    "paraphrase-multilingual-MiniLM-L12-v2": _make_sentence_transformer_factory(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "intfloat/multilingual-e5-base": _make_sentence_transformer_factory(
        "intfloat/multilingual-e5-base"
    ),
    "BAAI/bge-m3": _make_sentence_transformer_factory("BAAI/bge-m3"),
    "distiluse-base-multilingual-cased-v2": _make_sentence_transformer_factory(
        "distiluse-base-multilingual-cased-v2"
    ),
}

EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    "paraphrase-multilingual-MiniLM-L12-v2": EmbeddingModelSpec(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dimension=384,
    ),
    "intfloat/multilingual-e5-base": EmbeddingModelSpec(
        model_name="intfloat/multilingual-e5-base",
        embedding_dimension=768,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "BAAI/bge-m3": EmbeddingModelSpec(
        model_name="BAAI/bge-m3",
        embedding_dimension=1024,
    ),
}


def get_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    return EMBEDDING_MODEL_SPECS.get(
        model_name,
        EmbeddingModelSpec(model_name=model_name, embedding_dimension=0),
    )

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_ALIAS = os.getenv("QDRANT_COLLECTION_ALIAS", "steel_products_active")
DENSE_BATCH_SIZE = int(os.getenv("DENSE_BATCH_SIZE", "64"))
SOURCE_CANDIDATE_LIMIT = int(os.getenv("SOURCE_CANDIDATE_LIMIT", "300"))
TOP_K = 20
SOURCE_SCORE_HYBRID_WEIGHT = float(os.getenv("SOURCE_SCORE_HYBRID_WEIGHT", "0.55"))
SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT = float(
    os.getenv("SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT", "0.25")
)
SOURCE_SCORE_FIELD_WEIGHT = float(os.getenv("SOURCE_SCORE_FIELD_WEIGHT", "0.20"))
