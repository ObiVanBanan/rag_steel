from __future__ import annotations

import sys
import types

import numpy as np

from eval.embeddings import (
    EmbeddingTextAdapter,
    LocalSentenceTransformerEmbedder,
    get_eval_embedding_model_spec,
)


class _RoutingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, object]]] = []

    @staticmethod
    def _vectors(texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 0.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def encode_query(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(("query", list(texts), dict(kwargs)))
        return self._vectors(texts)

    def encode_document(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(("document", list(texts), dict(kwargs)))
        return self._vectors(texts)

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(("encode", list(texts), dict(kwargs)))
        return self._vectors(texts)


class _FallbackModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, object]]] = []

    @staticmethod
    def _vectors(texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 0.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(("encode", list(texts), dict(kwargs)))
        return self._vectors(texts)


def _make_local_embedder(model_name: str, model: object) -> LocalSentenceTransformerEmbedder:
    embedder = object.__new__(LocalSentenceTransformerEmbedder)
    embedder.model_name = model_name
    embedder.dimension = 2
    embedder.query_prefix = ""
    embedder.document_prefix = ""
    embedder.normalize_embeddings = True
    embedder.max_sequence_length = 512
    embedder.embedding_revision = ""
    embedder.embedding_dtype = "float32"
    embedder.embedding_device = "cpu"
    embedder._model = model
    return embedder


def test_transformers_fallback_does_not_expose_specialized_methods(monkeypatch) -> None:
    embeddings_mod = __import__("eval.embeddings", fromlist=["LocalSentenceTransformerEmbedder"])

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = object()
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    fake_torch.no_grad = lambda: _NoGrad()
    fake_torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(normalize=lambda *args, **kwargs: args[0])
    )

    class _FakeTokenizer:
        pass

    class _FakeModel:
        config = types.SimpleNamespace(hidden_size=2)

        def to(self, *_: object) -> None:
            return None

        def eval(self) -> None:
            return None

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    fake_transformers.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeModel()
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    embedder = object.__new__(LocalSentenceTransformerEmbedder)
    embedder.model_name = "intfloat/multilingual-e5-base"
    embedder.dimension = 2
    embedder.query_prefix = ""
    embedder.document_prefix = ""
    embedder.normalize_embeddings = True
    embedder.max_sequence_length = 512
    embedder.embedding_revision = ""
    embedder.embedding_dtype = "float32"
    embedder.embedding_device = "cpu"

    fallback_model = embeddings_mod.LocalSentenceTransformerEmbedder._load_transformers_fallback(
        embedder,
        model_source=None,
        revision=None,
    )

    assert hasattr(fallback_model, "encode")
    assert not hasattr(fallback_model, "encode_query")
    assert not hasattr(fallback_model, "encode_document")


def test_embedding_text_adapter_applies_e5_prefixes() -> None:
    adapter = EmbeddingTextAdapter("intfloat/multilingual-e5-base")

    assert adapter.prepare_query("Temper DN80 PN16") == "query: Temper DN80 PN16"
    assert adapter.prepare_document("SOURCE_SENTINEL") == "passage: SOURCE_SENTINEL"


def test_embedding_text_adapter_leaves_non_prefix_models_untouched() -> None:
    for model_name in [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "text-embedding-3-small",
    ]:
        adapter = EmbeddingTextAdapter(model_name)
        assert adapter.prepare_query("Temper DN80 PN16") == "Temper DN80 PN16"
        assert adapter.prepare_document("SOURCE_SENTINEL") == "SOURCE_SENTINEL"


def test_local_embedder_uses_specialized_query_and_document_methods() -> None:
    model = _RoutingModel()
    embedder = _make_local_embedder("intfloat/multilingual-e5-base", model)

    assert embedder.embed_query("Temper DN80 PN16") == [1.0, 0.0]
    assert embedder.embed_documents(["SOURCE_SENTINEL", "SECOND"]) == [[1.0, 0.0], [2.0, 0.0]]
    assert [call[0] for call in model.calls] == ["query", "document"]
    assert model.calls[0][1] == ["Temper DN80 PN16"]
    assert model.calls[1][1] == ["SOURCE_SENTINEL", "SECOND"]


def test_local_embedder_falls_back_to_prefixes_when_needed() -> None:
    model = _FallbackModel()
    embedder = _make_local_embedder("intfloat/multilingual-e5-base", model)

    assert embedder.embed_query("Temper DN80 PN16") == [1.0, 0.0]
    assert embedder.embed_documents(["SOURCE_SENTINEL"]) == [[1.0, 0.0]]
    assert model.calls[0][1] == ["query: Temper DN80 PN16"]
    assert model.calls[1][1] == ["passage: SOURCE_SENTINEL"]


def test_eval_model_specs_cover_the_supported_models() -> None:
    assert get_eval_embedding_model_spec("text-embedding-3-small").provider == "openai"
    assert get_eval_embedding_model_spec("BAAI/bge-m3").dimension == 1024
