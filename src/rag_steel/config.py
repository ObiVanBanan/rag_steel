"""Centralized runtime and indexing configuration for rag_steel."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


def resolve_embedding_device() -> str:
    explicit_device = os.getenv("EMBEDDING_DEVICE")
    if explicit_device:
        return explicit_device

    try:
        import torch
    except Exception:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    mps_backend = getattr(torch.backends, "mps", None)
    if sys.platform == "darwin" and mps_backend is not None:
        try:
            if mps_backend.is_available():
                return "mps"
        except Exception:
            pass

    return "cpu"


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    model_id: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    normalize_embeddings: bool = True
    max_sequence_length: int = 512
    preferred_dtype: str = "float32"


EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    "paraphrase-multilingual-MiniLM-L12-v2": EmbeddingModelSpec(
        model_id="paraphrase-multilingual-MiniLM-L12-v2",
        dimension=384,
        preferred_dtype="float32",
    ),
    "intfloat/multilingual-e5-base": EmbeddingModelSpec(
        model_id="intfloat/multilingual-e5-base",
        dimension=768,
        query_prefix="query: ",
        document_prefix="passage: ",
        preferred_dtype="float32",
    ),
    "BAAI/bge-m3": EmbeddingModelSpec(
        model_id="BAAI/bge-m3",
        dimension=1024,
        preferred_dtype="float16",
    ),
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
SUPPORTED_EMBEDDING_DTYPES = {"float16", "bfloat16", "float32"}


def get_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    return EMBEDDING_MODEL_SPECS.get(
        model_name,
        EmbeddingModelSpec(model_id=model_name, dimension=0),
    )


@dataclass(frozen=True, slots=True)
class Settings:
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    embedding_device: str
    embedding_dtype: str
    embedding_normalize: bool
    embedding_max_seq_length: int
    dense_batch_size: int
    qdrant_url: str
    qdrant_collection_alias: str
    qdrant_dense_vector_name: str
    qdrant_sparse_vector_name: str
    source_candidate_limit: int
    result_limit_default: int
    result_limit_max: int
    source_score_hybrid_weight: float
    source_score_text_exactness_weight: float
    source_score_field_weight: float

    def for_model(self, model_name: str) -> "Settings":
        spec = get_embedding_model_spec(model_name)
        return replace(
            self,
            embedding_model=model_name,
            embedding_dimension=spec.dimension or self.embedding_dimension,
            embedding_normalize=spec.normalize_embeddings,
            embedding_max_seq_length=spec.max_sequence_length,
            embedding_dtype=spec.preferred_dtype or self.embedding_dtype,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    default_spec = get_embedding_model_spec(
        os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    embedding_dtype = (
        os.getenv("EMBEDDING_DTYPE", default_spec.preferred_dtype).strip() or "float32"
    )
    if embedding_dtype not in SUPPORTED_EMBEDDING_DTYPES:
        raise ValueError(f"Unsupported embedding dtype: {embedding_dtype}")

    embedding_dimension = int(
        os.getenv("EMBEDDING_DIMENSION", str(default_spec.dimension or 0))
    )
    if embedding_dimension <= 0:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

    settings = Settings(
        embedding_model=default_spec.model_id,
        embedding_revision=os.getenv("EMBEDDING_REVISION", "").strip(),
        embedding_dimension=embedding_dimension,
        embedding_device=resolve_embedding_device(),
        embedding_dtype=embedding_dtype,
        embedding_normalize=_env_bool("EMBEDDING_NORMALIZE", default_spec.normalize_embeddings),
        embedding_max_seq_length=int(
            os.getenv("EMBEDDING_MAX_SEQ_LENGTH", str(default_spec.max_sequence_length))
        ),
        dense_batch_size=int(os.getenv("DENSE_BATCH_SIZE", "32")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection_alias=os.getenv("QDRANT_COLLECTION_ALIAS", "steel_products_active"),
        qdrant_dense_vector_name=os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense"),
        qdrant_sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
        source_candidate_limit=int(os.getenv("SOURCE_CANDIDATE_LIMIT", "300")),
        result_limit_default=int(os.getenv("RESULT_LIMIT_DEFAULT", "20")),
        result_limit_max=int(os.getenv("RESULT_LIMIT_MAX", "100")),
        source_score_hybrid_weight=float(os.getenv("SOURCE_SCORE_HYBRID_WEIGHT", "0.55")),
        source_score_text_exactness_weight=float(
            os.getenv("SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT", "0.25")
        ),
        source_score_field_weight=float(os.getenv("SOURCE_SCORE_FIELD_WEIGHT", "0.20")),
    )
    return settings


def _make_sentence_transformer_factory(model_name: str) -> Callable[[], Any]:
    def factory() -> Any:
        return load_embedding_model(get_settings().for_model(model_name))

    return factory


def _huggingface_cache_root() -> Path:
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cache_dir(model_name: str) -> Path:
    return _huggingface_cache_root() / f"models--{model_name.replace('/', '--')}"


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def resolve_cached_model_path(model_name: str, revision: str | None = None) -> Path | None:
    cache_dir = _model_cache_dir(model_name)
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidate_revisions: list[str] = []
    if revision:
        candidate_revisions.append(revision)
    main_ref = _read_text_file(cache_dir / "refs" / "main")
    if main_ref:
        candidate_revisions.append(main_ref)

    for candidate_revision in candidate_revisions:
        snapshot_path = snapshots_dir / candidate_revision
        if (snapshot_path / "config.json").exists():
            return snapshot_path

    snapshots = sorted(
        (
            path
            for path in snapshots_dir.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ),
        key=lambda path: path.name,
    )
    return snapshots[-1] if snapshots else None


class _TransformersEmbeddingModel:
    """SentenceTransformer-compatible fallback built directly on transformers."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        revision: str | None,
        max_seq_length: int,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = device
        self.max_seq_length = max_seq_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModel.from_pretrained(model_name, revision=revision)
        self._model.to(device)
        self._model.eval()

    def get_sentence_embedding_dimension(self) -> int:
        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Unable to determine embedding dimension from model config")
        return int(hidden_size)

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        del show_progress_bar

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.no_grad():
                outputs = self._model(**encoded)
            embeddings = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            if normalize_embeddings:
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
            vectors.extend(embedding.detach().cpu().numpy() for embedding in embeddings)
        if not vectors:
            return np.empty((0, self.get_sentence_embedding_dimension()), dtype=np.float32)
        array = np.asarray(vectors, dtype=np.float32)
        if convert_to_numpy:
            return array
        return array

    def encode_query(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def encode_document(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * expanded_mask).sum(dim=1)
        counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


def load_embedding_model(settings: Settings) -> Any:
    if settings.embedding_dtype not in SUPPORTED_EMBEDDING_DTYPES:
        raise ValueError(f"Unsupported embedding dtype: {settings.embedding_dtype}")

    revision = settings.embedding_revision or None
    model_source = resolve_cached_model_path(settings.embedding_model, revision)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception:
        model = _TransformersEmbeddingModel(
            model_name=str(model_source or settings.embedding_model),
            device=settings.embedding_device,
            revision=None if model_source else revision,
            max_seq_length=settings.embedding_max_seq_length,
        )
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        sentence_transformer_kwargs: dict[str, Any] = {
            "device": settings.embedding_device,
            "model_kwargs": {"dtype": dtype_map[settings.embedding_dtype]},
        }
        if model_source is not None:
            sentence_transformer_kwargs["local_files_only"] = True
        else:
            sentence_transformer_kwargs["revision"] = revision

        model = SentenceTransformer(
            str(model_source or settings.embedding_model),
            **sentence_transformer_kwargs,
        )
        model.max_seq_length = settings.embedding_max_seq_length

    actual_dimension = int(model.get_sentence_embedding_dimension())
    if actual_dimension != settings.embedding_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"configured={settings.embedding_dimension}, actual={actual_dimension}"
        )

    return model


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    "paraphrase-multilingual-MiniLM-L12-v2": _make_sentence_transformer_factory(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "intfloat/multilingual-e5-base": _make_sentence_transformer_factory(
        "intfloat/multilingual-e5-base"
    ),
    "BAAI/bge-m3": _make_sentence_transformer_factory("BAAI/bge-m3"),
}

DEFAULT_MODEL_NAME = get_settings().embedding_model
QDRANT_URL = get_settings().qdrant_url
QDRANT_COLLECTION_ALIAS = get_settings().qdrant_collection_alias
QDRANT_DENSE_VECTOR_NAME = get_settings().qdrant_dense_vector_name
QDRANT_SPARSE_VECTOR_NAME = get_settings().qdrant_sparse_vector_name
DENSE_BATCH_SIZE = get_settings().dense_batch_size
SOURCE_CANDIDATE_LIMIT = get_settings().source_candidate_limit
RESULT_LIMIT_DEFAULT = get_settings().result_limit_default
RESULT_LIMIT_MAX = get_settings().result_limit_max
TOP_K = RESULT_LIMIT_DEFAULT
SOURCE_SCORE_HYBRID_WEIGHT = get_settings().source_score_hybrid_weight
SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT = get_settings().source_score_text_exactness_weight
SOURCE_SCORE_FIELD_WEIGHT = get_settings().source_score_field_weight

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL_NAME",
    "DENSE_BATCH_SIZE",
    "EMBEDDING_MODEL_SPECS",
    "EmbeddingModelSpec",
    "MODEL_REGISTRY",
    "QDRANT_COLLECTION_ALIAS",
    "QDRANT_DENSE_VECTOR_NAME",
    "QDRANT_SPARSE_VECTOR_NAME",
    "QDRANT_URL",
    "RESULT_LIMIT_DEFAULT",
    "RESULT_LIMIT_MAX",
    "SOURCE_CANDIDATE_LIMIT",
    "SOURCE_SCORE_FIELD_WEIGHT",
    "SOURCE_SCORE_HYBRID_WEIGHT",
    "SOURCE_SCORE_TEXT_EXACTNESS_WEIGHT",
    "SUPPORTED_EMBEDDING_DTYPES",
    "Settings",
    "TOP_K",
    "_TransformersEmbeddingModel",
    "_make_sentence_transformer_factory",
    "get_embedding_model_spec",
    "get_settings",
    "load_embedding_model",
    "resolve_cached_model_path",
    "resolve_embedding_device",
]
