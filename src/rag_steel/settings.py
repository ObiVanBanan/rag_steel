"""Production runtime settings for rag_steel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DOTENV_VALUES: dict[str, str] = {}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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


def _env_float_or_none(name: str) -> float | None:
    value = _env_value(name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


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


_DOTENV_VALUES = load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    embedding_model: str
    embedding_dimension: int
    openai_api_key: str
    openai_base_url: str
    openai_timeout_seconds: float
    dense_batch_size: int
    qdrant_url: str
    qdrant_collection_alias: str
    qdrant_dense_vector_name: str
    qdrant_sparse_vector_name: str
    source_candidate_limit: int
    dense_score_threshold: float | None
    bm25_score_threshold: float | None
    result_limit_default: int
    result_limit_max: int


def get_settings() -> Settings:
    embedding_model = (
        _env_value("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
    ).strip() or DEFAULT_EMBEDDING_MODEL
    embedding_dimension = int(_env_value("EMBEDDING_DIMENSION", "1536") or "1536")
    if embedding_dimension <= 0:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

    return Settings(
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
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
        dense_score_threshold=_env_float_or_none("DENSE_SCORE_THRESHOLD"),
        bm25_score_threshold=_env_float_or_none("BM25_SCORE_THRESHOLD"),
        result_limit_default=int(_env_value("RESULT_LIMIT_DEFAULT", "20") or "20"),
        result_limit_max=int(_env_value("RESULT_LIMIT_MAX", "100") or "100"),
    )


QDRANT_URL = get_settings().qdrant_url
QDRANT_COLLECTION_ALIAS = get_settings().qdrant_collection_alias
QDRANT_DENSE_VECTOR_NAME = get_settings().qdrant_dense_vector_name
QDRANT_SPARSE_VECTOR_NAME = get_settings().qdrant_sparse_vector_name
DENSE_BATCH_SIZE = get_settings().dense_batch_size
SOURCE_CANDIDATE_LIMIT = get_settings().source_candidate_limit
RESULT_LIMIT_DEFAULT = get_settings().result_limit_default
RESULT_LIMIT_MAX = get_settings().result_limit_max


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DENSE_BATCH_SIZE",
    "QDRANT_COLLECTION_ALIAS",
    "QDRANT_DENSE_VECTOR_NAME",
    "QDRANT_SPARSE_VECTOR_NAME",
    "QDRANT_URL",
    "RESULT_LIMIT_DEFAULT",
    "RESULT_LIMIT_MAX",
    "SOURCE_CANDIDATE_LIMIT",
    "Settings",
    "get_settings",
    "load_dotenv",
]
