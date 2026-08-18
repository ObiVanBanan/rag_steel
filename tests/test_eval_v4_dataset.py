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


def test_build_v4_article_only_cases_do_not_leak_hard_or_soft_attrs() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records, _meta = build_v4_cases(documents, target_count=160)
    article_case = next(record for record in records if record.category == "article_only_exact")

    assert article_case.expected_attributes.raw_brand is None
    assert article_case.expected_attributes.resolved_brand is not None
    assert article_case.expected_attributes.article == article_case.query
    assert article_case.expected_attributes.dn is None
    assert article_case.expected_attributes.pn_bar is None
    assert article_case.expected_attributes.connection is None
    assert article_case.expected_attributes.body_material is None
    assert article_case.expected_attributes.medium is None
    assert article_case.expected_attributes.control is None
    assert article_case.expected_attributes.temperature is None
    assert article_case.expected_attributes.length_mm is None


def test_build_v4_conflict_cases_use_resolution_semantics() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
        SteelProductDocument(
            steel_id="b50",
            article="b50",
            article_norm="b50",
            article_compact="b50",
            name="ALSO DN80 PN16",
            name_variants=["ALSO DN80 PN16"],
            brand="ALSO",
            dn=80,
            pn_bar=16,
            connection="резьбовое",
            body_material="сталь 20",
            medium="газ",
            control="ручное",
            temperature="120 c",
            length_mm=None,
            semantic_text="ALSO DN80 PN16",
            lexical_text="ALSO DN80 PN16",
            ld_candidates=[],
        ),
        SteelProductDocument(
            steel_id="c90",
            article="c90",
            article_norm="c90",
            article_compact="c90",
            name="Broen DN100 PN25",
            name_variants=["Broen DN100 PN25"],
            brand="Broen",
            dn=100,
            pn_bar=25,
            connection="сварное",
            body_material="сталь 20",
            medium="газ",
            control="ручное",
            temperature="120 c",
            length_mm=None,
            semantic_text="Broen DN100 PN25",
            lexical_text="Broen DN100 PN25",
            ld_candidates=[],
        ),
    ]

    records, meta = build_v4_cases(documents, target_count=160)

    brand_conflict = next(
        record for record in records if record.category == "brand_article_conflict"
    )
    hard_conflict = next(record for record in records if record.category == "article_hard_conflict")

    assert brand_conflict.expected_attributes.raw_brand is not None
    assert (
        brand_conflict.expected_attributes.resolved_brand
        == brand_conflict.expected_attributes.raw_brand
    )
    assert brand_conflict.expected_attributes.resolved_article is None

    assert hard_conflict.expected_attributes.article is not None
    assert hard_conflict.expected_attributes.resolved_article is None
    assert hard_conflict.expected_attributes.dn not in {None, 100}
    assert hard_conflict.expected_attributes.pn_bar not in {None, 25}
    assert hard_conflict.expected_attributes.connection != "сварное"

    assert "source_brand_counts" in meta
