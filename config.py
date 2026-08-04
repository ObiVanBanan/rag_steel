"""Configuration for embedding and Qdrant indexing."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def _make_sentence_transformer_factory(model_name: str) -> Callable[[], Any]:
    def factory() -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    return factory


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    "all-MiniLM-L6-v2": _make_sentence_transformer_factory("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": _make_sentence_transformer_factory("all-mpnet-base-v2"),
    "paraphrase-multilingual-MiniLM-L12-v2": _make_sentence_transformer_factory(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "distiluse-base-multilingual-cased-v2": _make_sentence_transformer_factory(
        "distiluse-base-multilingual-cased-v2"
    ),
}

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
