from __future__ import annotations

from rag_steel.query_processor import EmbeddingTextAdapter, QueryProcessor, parse_query


def test_process_extracts_article_and_technical_fields() -> None:
    query = "Temper 1184399 DN80 PN16"
    processed = QueryProcessor().process(query)

    assert processed.raw == query
    assert processed.normalized.startswith("temper 1184399")
    assert processed.compact.startswith("temper1184399")
    assert "1184399" in processed.possible_article_tokens
    assert processed.brand == "Temper"
    assert processed.dn == 80.0
    assert processed.pn_bar == 16.0
    assert processed.connection is None
    assert processed.medium is None
    assert processed.control is None
    assert processed.semantic_text == "Temper 1184399, DN 80, PN 16"
    assert "temper 1184399" in processed.lexical_text
    assert "1184399" in processed.lexical_text
    assert "dn80" in processed.lexical_text
    assert "pn16" in processed.lexical_text
    assert "16" in processed.lexical_text
    assert not hasattr(processed, "route")
    assert parse_query(query).model_dump() == processed.model_dump()


def test_process_keeps_article_variants_for_punctuated_codes() -> None:
    query = "KSH.P.P.015.40-01"
    processed = QueryProcessor().process(query)

    assert processed.possible_article_tokens
    assert processed.semantic_text == query
    assert any("015.40-01" in token for token in processed.possible_article_tokens)
    assert any("0154001" in token for token in processed.possible_article_tokens)
    assert "0154001" in processed.lexical_text


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
