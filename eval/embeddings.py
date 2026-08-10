"""Evaluation-only embedding helpers for local and OpenAI backends."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from rag_steel.embeddings import Embedder, OpenAIEmbedder
from rag_steel.settings import get_settings

_DOTENV_VALUES: dict[str, str] = {}


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return _DOTENV_VALUES.get(name, default)


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


EVAL_EMBEDDING_MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
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


def get_eval_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    return EVAL_EMBEDDING_MODEL_SPECS.get(
        model_name,
        EmbeddingModelSpec(model_id=model_name, dimension=0),
    )


def _model_cache_dir(model_name: str) -> Path:
    hf_home = _env_value("HF_HOME")
    root = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    return root / f"models--{model_name.replace('/', '--')}"


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


class EmbeddingTextAdapter:
    def __init__(self, model_name: str) -> None:
        spec = get_eval_embedding_model_spec(model_name)
        self._query_prefix = spec.query_prefix
        self._document_prefix = spec.document_prefix

    def prepare_query(self, text: str) -> str:
        return f"{self._query_prefix}{text}" if self._query_prefix else text

    def prepare_document(self, text: str) -> str:
        return f"{self._document_prefix}{text}" if self._document_prefix else text


@dataclass(slots=True)
class LocalSentenceTransformerEmbedder:
    model_name: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    normalize_embeddings: bool = True
    max_sequence_length: int = 512
    embedding_revision: str = ""
    embedding_dtype: str = "float32"
    embedding_device: str = field(default_factory=resolve_embedding_device)
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        revision = self.embedding_revision or None
        model_source = resolve_cached_model_path(self.model_name, revision)

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._model = self._load_transformers_fallback(
                model_source=model_source,
                revision=revision,
            )
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            kwargs: dict[str, Any] = {
                "device": self.embedding_device,
                "model_kwargs": {"dtype": dtype_map[self.embedding_dtype]},
            }
            if model_source is not None:
                kwargs["local_files_only"] = True
            else:
                kwargs["revision"] = revision

            self._model = SentenceTransformer(str(model_source or self.model_name), **kwargs)
            self._model.max_seq_length = self.max_sequence_length

        actual_dimension = int(self.get_sentence_embedding_dimension())
        if actual_dimension != self.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"configured={self.dimension}, actual={actual_dimension}"
            )

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def _load_transformers_fallback(
        self,
        *,
        model_source: Path | None,
        revision: str | None,
    ) -> Any:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = self.model_name
        device = self.embedding_device
        max_sequence_length = self.max_sequence_length

        class _TransformersEmbeddingModel:
            def __init__(self) -> None:
                self._torch = torch
                self._device = device
                self.max_seq_length = max_sequence_length
                source = str(model_source or model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
                self._model = AutoModel.from_pretrained(source, revision=revision)
                self._model.to(self._device)
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
                    batch = list(texts[start : start + batch_size])
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
                        outputs.last_hidden_state, encoded["attention_mask"]
                    )
                    if normalize_embeddings:
                        embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    vectors.extend(embedding.detach().cpu().numpy() for embedding in embeddings)
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

        return _TransformersEmbeddingModel()

    def _encode(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        adapter = EmbeddingTextAdapter(self.model_name)
        prepared = (
            [adapter.prepare_query(text) for text in texts]
            if is_query
            else [adapter.prepare_document(text) for text in texts]
        )
        encode_fn = getattr(self._model, "encode_document", None) or self._model.encode
        encode_kwargs = {
            "batch_size": max(1, len(prepared)),
            "normalize_embeddings": self.normalize_embeddings,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        try:
            vectors = encode_fn(prepared, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("convert_to_numpy")
            vectors = encode_fn(prepared, **encode_kwargs)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [list(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, is_query=False)


def create_eval_embedder(model_name: str) -> Embedder:
    spec = get_eval_embedding_model_spec(model_name)
    if spec.provider == "openai":
        settings = get_settings()
        if settings.embedding_model != model_name or settings.embedding_dimension != spec.dimension:
            settings = replace(
                settings,
                embedding_model=model_name,
                embedding_dimension=spec.dimension,
            )
        return OpenAIEmbedder(settings)

    return LocalSentenceTransformerEmbedder(
        model_name=spec.model_id,
        dimension=spec.dimension,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        normalize_embeddings=spec.normalize_embeddings,
        max_sequence_length=spec.max_sequence_length,
        embedding_revision=_env_value("EMBEDDING_REVISION", "") or "",
        embedding_dtype=_env_value("EMBEDDING_DTYPE", spec.preferred_dtype) or spec.preferred_dtype,
    )


_DOTENV_VALUES = load_dotenv()


__all__ = [
    "EmbeddingModelSpec",
    "EmbeddingTextAdapter",
    "EVAL_EMBEDDING_MODEL_SPECS",
    "LocalSentenceTransformerEmbedder",
    "create_eval_embedder",
    "get_eval_embedding_model_spec",
    "load_dotenv",
    "resolve_cached_model_path",
    "resolve_embedding_device",
]
