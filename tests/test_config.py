from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _reload_config(*, disable_dotenv: bool = True):
    if disable_dotenv:
        os.environ["RAG_STEEL_DISABLE_DOTENV"] = "1"
    else:
        os.environ.pop("RAG_STEEL_DISABLE_DOTENV", None)
    sys.modules.pop("config", None)
    sys.modules.pop("rag_steel.config", None)
    return importlib.import_module("config")


def test_resolve_embedding_device_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda:1")
    config = _reload_config(disable_dotenv=False)

    assert config.resolve_embedding_device() == "cuda:1"


def test_resolve_embedding_device_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    config = _reload_config()

    assert config.resolve_embedding_device() == "cuda"


def test_bge_m3_is_production_default(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    config = _reload_config(disable_dotenv=False)

    settings = config.get_settings()
    spec = config.get_embedding_model_spec("text-embedding-3-small")

    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    assert spec.dimension == 1536
    assert spec.provider == "openai"
    assert spec.query_prefix == ""
    assert spec.document_prefix == ""
    assert spec.normalize_embeddings is True
    assert spec.preferred_dtype == "float32"


def test_sentence_transformer_factory_passes_revision_dtype_and_device(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("EMBEDDING_DTYPE", "float16")
    monkeypatch.setenv("EMBEDDING_MAX_SEQ_LENGTH", "512")
    monkeypatch.setenv("EMBEDDING_REVISION", "sha-123")
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float16="float16-token",
            bfloat16="bfloat16-token",
            float32="float32-token",
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=None),
        ),
    )

    captured: dict[str, object] = {}
    fake_module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(
            self,
            model_name: str,
            *,
            device: str,
            model_kwargs: dict[str, object],
            revision: str | None = None,
            local_files_only: bool | None = None,
        ) -> None:
            captured["model_name"] = model_name
            captured["revision"] = revision
            captured["device"] = device
            captured["model_kwargs"] = model_kwargs
            captured["local_files_only"] = local_files_only
            self.max_seq_length = 0

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    config = _reload_config(disable_dotenv=False)
    cached_model_path = config.resolve_cached_model_path("BAAI/bge-m3", "sha-123")
    factory = config._make_embedding_factory("BAAI/bge-m3")
    model = factory()

    assert model.get_sentence_embedding_dimension() == 1024
    assert model.max_seq_length == 512
    assert captured == {
        "model_name": str(cached_model_path or "BAAI/bge-m3"),
        "revision": None if cached_model_path else "sha-123",
        "device": "cpu",
        "model_kwargs": {"dtype": "float16-token"},
        "local_files_only": True if cached_model_path else None,
    }


def test_model_dimension_is_validated_after_loading(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("EMBEDDING_DTYPE", "float16")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float16="float16-token",
            bfloat16="bfloat16-token",
            float32="float32-token",
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=None),
        ),
    )

    fake_module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.max_seq_length = 0

        def get_sentence_embedding_dimension(self) -> int:
            return 768

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    config = _reload_config()

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        config.load_embedding_model(config.get_settings())


def test_sentence_transformer_factory_falls_back_to_transformers(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=None),
            no_grad=lambda: SimpleNamespace(
                __enter__=lambda self: None,
                __exit__=lambda self, exc_type, exc, tb: False,
            ),
            nn=SimpleNamespace(functional=SimpleNamespace(normalize=lambda x, p, dim: x)),
        ),
    )

    captured: dict[str, object] = {}

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise RuntimeError("broken optional dependency stack")
        return original_import(name, globals, locals, fromlist, level)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, revision: str | None = None) -> object:
            captured["tokenizer_model_name"] = model_name
            captured["tokenizer_revision"] = revision
            return object()

    class FakeModel:
        config = SimpleNamespace(hidden_size=1024)

        @classmethod
        def from_pretrained(
            cls, model_name: str, revision: str | None = None
        ) -> "FakeModel":
            captured["transformers_model_name"] = model_name
            captured["transformers_revision"] = revision
            return cls()

        def to(self, device: str) -> None:
            captured["device"] = device

        def eval(self) -> None:
            captured["eval_called"] = True

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import builtins

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", raising_import)

    config = _reload_config()
    cached_model_path = config.resolve_cached_model_path("BAAI/bge-m3")
    factory = config._make_embedding_factory("BAAI/bge-m3")
    model = factory()

    assert model.get_sentence_embedding_dimension() == 1024
    assert captured == {
        "tokenizer_model_name": str(cached_model_path or "BAAI/bge-m3"),
        "tokenizer_revision": None,
        "transformers_model_name": str(cached_model_path or "BAAI/bge-m3"),
        "transformers_revision": None,
        "device": "cpu",
        "eval_called": True,
    }


def test_resolve_cached_model_path_prefers_main_ref(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "hub"
    model_dir = cache_root / "models--BAAI--bge-m3"
    snapshot = model_dir / "snapshots" / "sha-main"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("sha-main", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    config = _reload_config()

    assert config.resolve_cached_model_path("BAAI/bge-m3") == snapshot


def test_openai_embedding_factory_uses_httpx_client(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12")

    captured: dict[str, object] = {}
    fake_module = ModuleType("httpx")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [1.0] * 1536}]}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            captured["path"] = path
            captured["payload"] = json
            return FakeResponse()

    fake_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "httpx", fake_module)

    config = _reload_config()
    model = config.load_embedding_model(config.get_settings())
    embeddings = model.encode(
        ["alpha"],
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    assert model.get_sentence_embedding_dimension() == 1536
    assert embeddings.shape == (1, 1536)
    assert captured["path"] == "embeddings"
    assert captured["payload"] == {
        "input": ["alpha"],
        "model": "text-embedding-3-small",
        "encoding_format": "float",
    }
    assert captured["client_kwargs"] == {
        "base_url": "https://example.invalid/v1/",
        "headers": {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        "timeout": 12.0,
    }


def test_openai_embedding_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _reload_config()

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        config.load_embedding_model(config.get_settings())


def test_load_dotenv_populates_missing_environment_values(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-dotenv-key",
                "EMBEDDING_PROVIDER=openai",
                "EMBEDDING_MODEL=text-embedding-3-small",
                "EMBEDDING_DIMENSION=1536",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_STEEL_ENV_FILE", str(env_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)

    config = _reload_config(disable_dotenv=False)
    settings = config.get_settings()

    assert settings.openai_api_key == "test-dotenv-key"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536


def test_load_dotenv_does_not_override_existing_environment_values(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("RAG_STEEL_ENV_FILE", str(env_file))
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    config = _reload_config()
    settings = config.get_settings()

    assert settings.openai_api_key == "from-env"
