"""Compatibility shim for the split settings/embeddings modules."""

from __future__ import annotations

from rag_steel.embeddings import Embedder, OpenAIEmbedder, create_embedder
from rag_steel.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DENSE_BATCH_SIZE,
    QDRANT_COLLECTION_ALIAS,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_URL,
    RESULT_LIMIT_DEFAULT,
    RESULT_LIMIT_MAX,
    SOURCE_CANDIDATE_LIMIT,
    Settings,
    get_settings,
    load_dotenv,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DENSE_BATCH_SIZE",
    "Embedder",
    "OpenAIEmbedder",
    "QDRANT_COLLECTION_ALIAS",
    "QDRANT_DENSE_VECTOR_NAME",
    "QDRANT_SPARSE_VECTOR_NAME",
    "QDRANT_URL",
    "RESULT_LIMIT_DEFAULT",
    "RESULT_LIMIT_MAX",
    "SOURCE_CANDIDATE_LIMIT",
    "Settings",
    "create_embedder",
    "get_settings",
    "load_dotenv",
]
