from __future__ import annotations

from rag_steel.query_processor import EmbeddingTextAdapter, QueryProcessor, parse_query


def test_process_extracts_article_and_technical_fields() -> None:
    processed = QueryProcessor().process("Temper 1184399 Ду80 Ру16")

    assert processed.raw == "Temper 1184399 Ду80 Ру16"
    assert processed.normalized == "temper 1184399 ду80 ру16"
    assert processed.compact == "temper1184399ду80ру16"
    assert processed.possible_article_tokens == ["1184399"]
    assert processed.brand == "Temper"
    assert processed.dn == 80.0
    assert processed.pn_bar == 16.0
    assert processed.connection is None
    assert processed.medium is None
    assert processed.control is None
    assert processed.semantic_text == "query: Temper 1184399, DN 80, PN 16"
    assert "temper 1184399 ду80 ру16" in processed.lexical_text
    assert "1184399" in processed.lexical_text
    assert "dn80" in processed.lexical_text
    assert "ду80" in processed.lexical_text
    assert "pn16" in processed.lexical_text
    assert "ру16" in processed.lexical_text
    assert "16 бар" in processed.lexical_text
    assert not hasattr(processed, "route")
    assert parse_query("Temper 1184399 Ду80 Ру16").model_dump() == processed.model_dump()


def test_process_keeps_article_variants_for_punctuated_codes() -> None:
    processed = QueryProcessor().process("КШ.П.П.015.40-01")

    assert processed.possible_article_tokens == [
        "КШ.П.П.015.40-01",
        "кш.п.п.015.40-01",
        "кшпп0154001",
    ]
    assert processed.semantic_text == "query: КШ.П.П.015.40-01"
    assert processed.lexical_text.startswith("кш.п.п.015.40-01")
    assert "КШ.П.П.015.40-01" in processed.lexical_text
    assert "кшпп0154001" in processed.lexical_text


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
