"""Centralized runtime and indexing configuration for rag_steel."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

_DOTENV_VALUES: dict[str, str] = {}


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return _DOTENV_VALUES.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _find_dotenv_path() -> Path | None:
    env_path = os.getenv("RAG_STEEL_ENV_FILE", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        return candidate if candidate.is_file() else None

    for base_dir in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = base_dir / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv() -> dict[str, str]:
    if os.getenv("RAG_STEEL_DISABLE_DOTENV", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {}

    dotenv_path = _find_dotenv_path()
    if dotenv_path is None:
        return {}

    try:
        loaded: dict[str, str] = {}
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            loaded.setdefault(key, value)
    except Exception:
        return {}
    return loaded

def resolve_embedding_device() -> str:
    explicit_device = _env_value("EMBEDDING_DEVICE")
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
    provider: str = "local"
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
    "text-embedding-3-small": EmbeddingModelSpec(
        model_id="text-embedding-3-small",
        dimension=1536,
        provider="openai",
        preferred_dtype="float32",
        max_sequence_length=8191,
    ),
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_EMBEDDING_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_EMBEDDING_PROVIDERS = {"local", "openai"}


def get_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    return EMBEDDING_MODEL_SPECS.get(
        model_name,
        EmbeddingModelSpec(model_id=model_name, dimension=0),
    )


@dataclass(frozen=True, slots=True)
class Settings:
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    embedding_device: str
    embedding_dtype: str
    embedding_normalize: bool
    embedding_max_seq_length: int
    openai_api_key: str
    openai_base_url: str
    openai_timeout_seconds: float
    dense_batch_size: int
    qdrant_url: str
    qdrant_collection_alias: str
    qdrant_dense_vector_name: str
    qdrant_sparse_vector_name: str
    source_candidate_limit: int
    result_limit_default: int
    result_limit_max: int

    def for_model(self, model_name: str) -> "Settings":
        spec = get_embedding_model_spec(model_name)
        return replace(
            self,
            embedding_provider=spec.provider,
            embedding_model=model_name,
            embedding_dimension=spec.dimension or self.embedding_dimension,
            embedding_normalize=spec.normalize_embeddings,
            embedding_max_seq_length=spec.max_sequence_length,
            embedding_dtype=spec.preferred_dtype or self.embedding_dtype,
        )

_DOTENV_VALUES = load_dotenv()


def get_settings() -> Settings:
    default_spec = get_embedding_model_spec(
        _env_value("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
    )
    embedding_provider = (
        _env_value("EMBEDDING_PROVIDER", default_spec.provider) or "local"
    ).strip() or "local"
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {embedding_provider}")

    embedding_dtype = (
        (_env_value("EMBEDDING_DTYPE", default_spec.preferred_dtype) or "float32").strip()
        or "float32"
    )
    if embedding_dtype not in SUPPORTED_EMBEDDING_DTYPES:
        raise ValueError(f"Unsupported embedding dtype: {embedding_dtype}")

    embedding_dimension = int(
        _env_value("EMBEDDING_DIMENSION", str(default_spec.dimension or 0))
        or str(default_spec.dimension or 0)
    )
    if embedding_dimension <= 0:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

    settings = Settings(
        embedding_provider=embedding_provider,
        embedding_model=default_spec.model_id,
        embedding_revision=(_env_value("EMBEDDING_REVISION", "") or "").strip(),
        embedding_dimension=embedding_dimension,
        embedding_device=resolve_embedding_device(),
        embedding_dtype=embedding_dtype,
        embedding_normalize=_env_bool("EMBEDDING_NORMALIZE", default_spec.normalize_embeddings),
        embedding_max_seq_length=int(
            _env_value("EMBEDDING_MAX_SEQ_LENGTH", str(default_spec.max_sequence_length))
            or str(default_spec.max_sequence_length)
        ),
        openai_api_key=(_env_value("OPENAI_API_KEY", "") or "").strip(),
        openai_base_url=(
            (_env_value("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").strip()
            or "https://api.openai.com/v1"
        ),
        openai_timeout_seconds=float(_env_value("OPENAI_TIMEOUT_SECONDS", "60") or "60"),
        dense_batch_size=int(_env_value("DENSE_BATCH_SIZE", "32") or "32"),
        qdrant_url=_env_value("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333",
        qdrant_collection_alias=(
            _env_value("QDRANT_COLLECTION_ALIAS", "steel_products_active")
            or "steel_products_active"
        ),
        qdrant_dense_vector_name=_env_value("QDRANT_DENSE_VECTOR_NAME", "dense") or "dense",
        qdrant_sparse_vector_name=_env_value("QDRANT_SPARSE_VECTOR_NAME", "sparse") or "sparse",
        source_candidate_limit=int(_env_value("SOURCE_CANDIDATE_LIMIT", "300") or "300"),
        result_limit_default=int(_env_value("RESULT_LIMIT_DEFAULT", "20") or "20"),
        result_limit_max=int(_env_value("RESULT_LIMIT_MAX", "100") or "100"),
    )
    return settings


def _make_embedding_factory(model_name: str) -> Callable[[], Any]:
    def factory() -> Any:
        return load_embedding_model(get_settings().for_model(model_name))

    return factory


def _huggingface_cache_root() -> Path:
    hf_home = _env_value("HF_HOME")
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


class _OpenAIEmbeddingModel:
    """Batch embedding client backed by the OpenAI embeddings API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        import httpx

        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.openai_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.openai_timeout_seconds,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return int(self._settings.embedding_dimension)

    def _embed_batch(self, batch: list[str], dimensions: int | None) -> np.ndarray:
        payload: dict[str, Any] = {
            "input": batch,
            "model": self._settings.embedding_model,
            "encoding_format": "float",
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = self._client.post("embeddings", json=payload)
        response.raise_for_status()
        response_payload = response.json()
        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings response is missing data")
        vectors = [item.get("embedding") for item in data]
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise RuntimeError(f"Embeddings must be 2D, got shape {array.shape}")
        return array

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

        if not texts:
            return np.empty((0, self.get_sentence_embedding_dimension()), dtype=np.float32)

        vectors: list[np.ndarray] = []
        requested_dimensions = None
        spec = get_embedding_model_spec(self._settings.embedding_model)
        if spec.dimension and self._settings.embedding_dimension != spec.dimension:
            requested_dimensions = self._settings.embedding_dimension

        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            embeddings = self._embed_batch(batch, requested_dimensions)
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.clip(norms, 1e-12, None)
                embeddings = embeddings / norms
            vectors.append(embeddings)

        array = np.vstack(vectors).astype(np.float32, copy=False)
        if convert_to_numpy:
            return array
        return array

    def encode_query(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(texts, **kwargs)

    def encode_document(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(texts, **kwargs)


def load_embedding_model(settings: Settings) -> Any:
    if settings.embedding_provider == "openai":
        model = _OpenAIEmbeddingModel(settings)
        actual_dimension = int(model.get_sentence_embedding_dimension())
        if actual_dimension != settings.embedding_dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"configured={settings.embedding_dimension}, actual={actual_dimension}"
            )
        return model

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
    "paraphrase-multilingual-MiniLM-L12-v2": _make_embedding_factory(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "intfloat/multilingual-e5-base": _make_embedding_factory(
        "intfloat/multilingual-e5-base"
    ),
    "BAAI/bge-m3": _make_embedding_factory("BAAI/bge-m3"),
    "text-embedding-3-small": _make_embedding_factory("text-embedding-3-small"),
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
    "SUPPORTED_EMBEDDING_DTYPES",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "Settings",
    "TOP_K",
    "_OpenAIEmbeddingModel",
    "_TransformersEmbeddingModel",
    "_make_embedding_factory",
    "get_embedding_model_spec",
    "get_settings",
    "load_embedding_model",
    "resolve_cached_model_path",
    "resolve_embedding_device",
]
