from __future__ import annotations

from eval.build_v4_eval_dataset import _pn_minimum_semantics_cases, build_v4_cases
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(*, article: str, pn_bar: float) -> SteelProductDocument:
    return SteelProductDocument(
        steel_id=article,
        article=article,
        article_norm=article.casefold(),
        article_compact=article.casefold(),
        name=f"Temper DN50 PN{int(pn_bar)}",
        name_variants=[f"Temper DN50 PN{int(pn_bar)}"],
        brand="Temper",
        dn=50,
        pn_bar=pn_bar,
        connection="flanged",
        body_material="сталь 20",
        medium="газ",
        control="ручное",
        temperature="120 c",
        length_mm=None,
        semantic_text=f"Temper DN50 PN{int(pn_bar)}",
        lexical_text=f"Temper DN50 PN{int(pn_bar)}",
        ld_candidates=[
            LDProduct(
                article=f"ld-{article}",
                article_norm=f"ld-{article}".casefold(),
                name=f"LD {article}",
            )
        ],
    )


def test_pn_minimum_semantics_cases_include_higher_pn_candidates() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    cases = _pn_minimum_semantics_cases(documents, count=10)

    assert [case.query for case in cases] == ["Temper DN50 PN16", "Temper DN50 PN25"]
    assert cases[0].expected_attributes.pn_bar == 16
    assert cases[1].expected_attributes.pn_bar == 25
    assert set(cases[0].eligible_competitor_articles) == {"a16", "a25", "a40"}
    assert set(cases[1].eligible_competitor_articles) == {"a25", "a40"}
    assert all(case.category == "pn_minimum_semantics" for case in cases)


def test_build_v4_cases_includes_pn_minimum_semantics_category() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records, meta = build_v4_cases(documents, target_count=160)

    assert any(record.category == "pn_minimum_semantics" for record in records)
    assert meta["by_category"]["pn_minimum_semantics"] >= 1
    assert meta["by_brand"]["Temper"] >= 1
