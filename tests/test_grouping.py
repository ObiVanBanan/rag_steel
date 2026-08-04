from __future__ import annotations

from hashlib import sha1

import pandas as pd

from rag_steel.data_builder import build_source_documents_from_frame
from rag_steel.normalization import (
    normalize_article,
    normalize_connection,
    normalize_control,
    normalize_text,
)


def _make_grouping_frame() -> pd.DataFrame:
    rows = [
        {
            "ld_name": "Кран шаровой LD Ду80 Ру16",
            "ld_article": "11100800162MULD000003000",
            "ld_url": "https://ld.example/1",
            "ld_dn": 80,
            "ld_pn_mpa": 1.6,
            "ld_connection": "фланцевое",
            "ld_medium": "жидкость",
            "ld_control": "ручное",
            "ld_temp": None,
            "ld_length": 300,
            "steel_name": "Кран шаровой Temper Ду80 Ру16",
            "steel_article": "КШ.П.П.015.40-01",
            "steel_url": "https://steel.example/a",
            "steel_dn": 80,
            "steel_pn_bar": 16,
            "steel_connection": "фланцевый",
            "steel_medium": "жидкость",
            "steel_control": "ручное",
            "steel_temp": "до +80",
            "steel_length": "300 мм",
            "match_score": 8,
            "match_max": 7,
            "price_ld": 12130,
        },
        {
            "ld_name": "Кран шаровой LD Ду80 Ру16",
            "ld_article": "11100800162MULD000003000",
            "ld_url": "https://ld.example/1",
            "ld_dn": 80,
            "ld_pn_mpa": 1.6,
            "ld_connection": "фланцевое",
            "ld_medium": "жидкость",
            "ld_control": "ручное",
            "ld_temp": None,
            "ld_length": 300,
            "steel_name": "Кран шаровой Temper Ду80 Ру16",
            "steel_article": "КШ.П.П.015.40-01",
            "steel_url": "https://steel.example/a",
            "steel_dn": 80,
            "steel_pn_bar": 16,
            "steel_connection": "фланцевый",
            "steel_medium": "жидкость",
            "steel_control": "ручное",
            "steel_temp": "до +80",
            "steel_length": "300 мм",
            "match_score": 8,
            "match_max": 7,
            "price_ld": 12130,
        },
        {
            "ld_name": "Кран шаровой LD Ду50 Ру16",
            "ld_article": "11100800162MULD000004000",
            "ld_url": "https://ld.example/2",
            "ld_dn": 50,
            "ld_pn_mpa": 1.6,
            "ld_connection": "фланцевое",
            "ld_medium": "жидкость",
            "ld_control": "ручное",
            "ld_temp": None,
            "ld_length": 250,
            "steel_name": "Кран Шаровой Temper Ду80 Ру16",
            "steel_article": "КШ.П.П.015.40-01",
            "steel_url": "https://steel.example/a",
            "steel_dn": 80,
            "steel_pn_bar": 16,
            "steel_connection": "фланцевый",
            "steel_medium": "жидкость",
            "steel_control": "ручное",
            "steel_temp": "до +80",
            "steel_length": "300 мм",
            "match_score": 7,
            "match_max": 7,
            "price_ld": 9800,
        },
        {
            "ld_name": "Кран шаровой LD Ду50 Ру16",
            "ld_article": "11100800162MULD000004000",
            "ld_url": "https://ld.example/2",
            "ld_dn": 50,
            "ld_pn_mpa": 1.6,
            "ld_connection": "фланцевое",
            "ld_medium": "жидкость",
            "ld_control": "ручное",
            "ld_temp": None,
            "ld_length": 250,
            "steel_name": "Кран Шаровой Temper Ду50 Ру16",
            "steel_article": "КШ.П.П.015.40-01",
            "steel_url": "https://steel.example/b",
            "steel_dn": 50,
            "steel_pn_bar": 16,
            "steel_connection": "фланцевый",
            "steel_medium": "жидкость",
            "steel_control": "ручное",
            "steel_temp": "до +80",
            "steel_length": "250 мм",
            "match_score": 9,
            "match_max": 7,
            "price_ld": 9800,
        },
        {
            "ld_name": "Кран шаровой LD Ду80 Ру16",
            "ld_article": "99900000000MULD000003000",
            "ld_url": "https://ld.example/3",
            "ld_dn": 80,
            "ld_pn_mpa": 1.6,
            "ld_connection": "фланцевое",
            "ld_medium": "жидкость",
            "ld_control": "ручное",
            "ld_temp": None,
            "ld_length": 300,
            "steel_name": "а0486",
            "steel_article": "а0486",
            "steel_url": "https://steel.example/c",
            "steel_dn": 80,
            "steel_pn_bar": 16,
            "steel_connection": "фланцевый",
            "steel_medium": "жидкость",
            "steel_control": "ручное",
            "steel_temp": "до +80",
            "steel_length": "300 мм",
            "match_score": 8,
            "match_max": 7,
            "price_ld": 11111,
        },
    ]
    return pd.DataFrame(rows)


def test_build_source_documents_groups_rows_and_deduplicates_ld_candidates() -> None:
    docs = build_source_documents_from_frame(_make_grouping_frame())

    assert [doc.steel_id for doc in docs] == sorted(doc.steel_id for doc in docs)
    assert len(docs) == 3
    assert len({doc.steel_id for doc in docs}) == 3

    first_doc = next(
        doc for doc in docs if doc.article_norm == "кш.п.п.015.40-01" and doc.dn == 80.0
    )
    assert first_doc.name == "Кран Шаровой Temper Ду80 Ру16"
    assert first_doc.name_variants == [
        "Кран Шаровой Temper Ду80 Ру16",
        "Кран шаровой Temper Ду80 Ру16",
    ]
    assert [candidate.article_norm for candidate in first_doc.ld_candidates] == [
        "11100800162muld000003000",
        "11100800162muld000004000",
    ]
    assert len({candidate.article_norm for candidate in first_doc.ld_candidates}) == len(
        first_doc.ld_candidates
    )
    assert first_doc.dn == 80.0
    assert first_doc.pn_bar == 16.0
    assert first_doc.connection == "фланцевое"
    assert first_doc.control == "ручное"

    second_doc = next(
        doc for doc in docs if doc.article_norm == "кш.п.п.015.40-01" and doc.dn == 50.0
    )
    assert second_doc.ld_candidates[0].article_norm == "11100800162muld000004000"
    assert second_doc.name_variants == ["Кран Шаровой Temper Ду50 Ру16"]


def test_build_source_documents_is_order_independent() -> None:
    frame = _make_grouping_frame()
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)

    docs_a = build_source_documents_from_frame(frame)
    docs_b = build_source_documents_from_frame(shuffled)

    assert [doc.model_dump() for doc in docs_a] == [doc.model_dump() for doc in docs_b]


def test_build_source_documents_populates_search_texts() -> None:
    doc = next(
        item
        for item in build_source_documents_from_frame(_make_grouping_frame())
        if item.article == "КШ.П.П.015.40-01" and item.dn == 80.0
    )

    semantic_text = doc.semantic_text
    lexical_text = doc.lexical_text
    article_norm = normalize_article(doc.article).article_norm or ""
    article_compact = normalize_article(doc.article).article_compact or ""
    dn_text = f"{int(doc.dn or 0)}"
    pn_text = f"{int(doc.pn_bar or 0)}"

    assert doc.name in semantic_text
    assert doc.brand in semantic_text
    assert f"DN {dn_text}" in semantic_text
    assert f"PN {pn_text} бар" in semantic_text
    assert f"Соединение: {doc.connection}" in semantic_text
    assert f"Рабочая среда: {doc.medium}" in semantic_text
    assert f"Управление: {doc.control}" in semantic_text
    assert f"Температура: {doc.temperature}" in semantic_text
    assert f"Длина: {int(doc.length_mm or 0)} мм" in semantic_text
    assert semantic_text.rstrip().endswith(f"Артикул: {doc.article}")
    assert all(candidate.name not in semantic_text for candidate in doc.ld_candidates)
    assert all(candidate.article not in semantic_text for candidate in doc.ld_candidates)

    assert doc.name in lexical_text
    assert doc.name_variants[0] in lexical_text
    assert doc.name_variants[1] in lexical_text
    assert doc.brand in lexical_text
    assert doc.article in lexical_text
    assert article_norm in lexical_text
    assert article_compact in lexical_text
    assert f"DN{dn_text}" in lexical_text
    assert f"DN {dn_text}" in lexical_text
    assert f"Ду{dn_text}" in lexical_text
    assert f"Ду {dn_text}" in lexical_text
    assert f"PN{pn_text}" in lexical_text
    assert f"PN {pn_text}" in lexical_text
    assert f"Ру{pn_text}" in lexical_text
    assert f"Ру {pn_text}" in lexical_text
    assert f"{pn_text} бар" in lexical_text
    assert doc.connection in lexical_text
    assert doc.medium in lexical_text
    assert doc.control in lexical_text


def test_stable_id_matches_document_formula() -> None:
    doc = build_source_documents_from_frame(_make_grouping_frame())[0]
    expected = sha1(
        (
            doc.article_compact
            + normalize_text(doc.name)
            + f"{doc.dn:g}"
            + f"{doc.pn_bar:g}"
            + normalize_connection(doc.connection)
            + normalize_control(doc.control)
        ).encode("utf-8")
    ).hexdigest()

    assert doc.steel_id == expected
