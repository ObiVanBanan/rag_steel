from __future__ import annotations

from eval.embeddings import EmbeddingTextAdapter, get_eval_embedding_model_spec


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


def test_eval_model_specs_cover_the_supported_models() -> None:
    assert get_eval_embedding_model_spec("text-embedding-3-small").provider == "openai"
    assert get_eval_embedding_model_spec("BAAI/bge-m3").dimension == 1024
