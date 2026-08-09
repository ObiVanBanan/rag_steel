from __future__ import annotations

from rag_steel.embedding_text import EmbeddingTextAdapter


def test_embedding_adapter_handles_e5_prefixes() -> None:
    adapter = EmbeddingTextAdapter("intfloat/multilingual-e5-base")

    assert adapter.prepare_query("temper 1184399") == "query: temper 1184399"
    assert adapter.prepare_query("query: temper 1184399") == "query: temper 1184399"
    assert adapter.prepare_document("temper 1184399") == "passage: temper 1184399"
    assert adapter.prepare_document("passage: temper 1184399") == "passage: temper 1184399"


def test_embedding_adapter_leaves_non_e5_texts_untouched() -> None:
    adapter = EmbeddingTextAdapter("paraphrase-multilingual-MiniLM-L12-v2")

    assert adapter.prepare_query("temper 1184399") == "temper 1184399"
    assert adapter.prepare_document("temper 1184399") == "temper 1184399"


def test_embedding_adapter_leaves_bge_m3_texts_untouched() -> None:
    adapter = EmbeddingTextAdapter("BAAI/bge-m3")

    assert adapter.prepare_query("Temper DN80 PN16") == "Temper DN80 PN16"
    assert adapter.prepare_document("Temper DN80 PN16") == "Temper DN80 PN16"
