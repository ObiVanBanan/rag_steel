"""Конфигурация и тестирование разных sentence-transformers."""

from sentence_transformers import SentenceTransformer
from typing import Dict, Callable

# Тестируемые модели (название → фабрика)
MODEL_REGISTRY: Dict[str, Callable[[], SentenceTransformer]] = {
    "all-MiniLM-L6-v2": lambda: SentenceTransformer("all-MiniLM-L6-v2"),   # 384 dim, быстрая
    "all-mpnet-base-v2": lambda: SentenceTransformer("all-mpnet-base-v2"),  # 768 dim, точнее
    "paraphrase-multilingual-MiniLM-L12-v2": lambda: SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),  # 384 dim, лучше для русского
    "distiluse-base-multilingual-cased-v2": lambda: SentenceTransformer(
        "distiluse-base-multilingual-cased-v2"
    ),  # 512 dim, мультиязычная
}

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ld_analogs_hybrid"
TOP_K = 20