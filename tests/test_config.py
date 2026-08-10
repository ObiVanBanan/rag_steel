from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


def _reload_settings(*, disable_dotenv: bool = True):
    if disable_dotenv:
        os.environ["RAG_STEEL_DISABLE_DOTENV"] = "1"
    else:
        os.environ.pop("RAG_STEEL_DISABLE_DOTENV", None)
    sys.modules.pop("rag_steel.settings", None)
    return importlib.import_module("rag_steel.settings")


def _reload_embeddings():
    sys.modules.pop("rag_steel.embeddings", None)
    return importlib.import_module("rag_steel.embeddings")


def test_get_settings_uses_production_defaults_and_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("DENSE_SCORE_THRESHOLD", "0.75")
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "4.0")

    settings_mod = _reload_settings(disable_dotenv=False)
    settings = settings_mod.get_settings()

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    assert settings.dense_score_threshold == 0.75
    assert settings.bm25_score_threshold == 4.0
    assert settings.qdrant_collection_alias == "steel_products_active"


def test_load_dotenv_populates_missing_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-dotenv-key",
                "EMBEDDING_MODEL=text-embedding-3-small",
                "EMBEDDING_DIMENSION=1536",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_STEEL_ENV_FILE", str(env_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)

    settings_mod = _reload_settings(disable_dotenv=False)
    settings = settings_mod.get_settings()

    assert settings.openai_api_key == "test-dotenv-key"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536


def test_openai_embedder_uses_httpx_client(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12")

    settings_mod = _reload_settings()
    embeddings_mod = _reload_embeddings()
    settings = settings_mod.get_settings()

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [1.0] * 1024}, {"embedding": [2.0] * 1024}]}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            captured["path"] = path
            captured["payload"] = json
            item_count = len(json["input"])

            class _BatchResponse(FakeResponse):
                def json(self) -> dict[str, object]:
                    return {
                        "data": [
                            {"embedding": [float(index + 1)] * 1024}
                            for index in range(item_count)
                        ]
                    }

            return _BatchResponse()

    monkeypatch.setattr(embeddings_mod.httpx, "Client", FakeClient)

    embedder = embeddings_mod.create_embedder(settings)
    query_vector = embedder.embed_query("alpha")
    document_vectors = embedder.embed_documents(["alpha", "beta"])

    assert embedder.model_name == "text-embedding-3-small"
    assert len(query_vector) == 1024
    assert len(document_vectors) == 2
    assert len(document_vectors[0]) == 1024
    assert captured["path"] == "embeddings"
    assert captured["payload"] == {
        "input": ["alpha", "beta"],
        "model": "text-embedding-3-small",
        "encoding_format": "float",
        "dimensions": 1024,
    }
    assert captured["client_kwargs"] == {
        "base_url": "https://example.invalid/v1/",
        "headers": {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        "timeout": 12.0,
    }


def _make_openai_embedder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dimension: int,
    response_data: list[dict[str, object]],
):
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", str(dimension))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12")

    settings_mod = _reload_settings()
    embeddings_mod = _reload_embeddings()
    settings = settings_mod.get_settings()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": response_data}

    class FakeClient:
        def __init__(self, **_: object) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(embeddings_mod.httpx, "Client", FakeClient)
    return embeddings_mod.create_embedder(settings)


def test_openai_embedder_rejects_wrong_vector_count(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _make_openai_embedder(
        monkeypatch,
        dimension=1024,
        response_data=[{"embedding": [1.0] * 1024}],
    )

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        embedder.embed_documents(["alpha", "beta"])


def test_openai_embedder_rejects_wrong_vector_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = _make_openai_embedder(
        monkeypatch,
        dimension=1024,
        response_data=[
            {"embedding": [1.0] * 512},
            {"embedding": [2.0] * 512},
        ],
    )

    with pytest.raises(RuntimeError, match="dimension 512, expected 1024"):
        embedder.embed_documents(["alpha", "beta"])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_openai_embedder_rejects_non_finite_values(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    embedder = _make_openai_embedder(
        monkeypatch,
        dimension=1024,
        response_data=[{"embedding": [bad_value] + [1.0] * 1023}],
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        embedder.embed_query("alpha")
