from __future__ import annotations

from eval.build_v4_eval_dataset import (
    _pn_minimum_semantics_cases,
    _regression_cases,
    build_v4_cases,
    find_v4_eligible_documents,
)
from rag_steel.normalization import normalize_article
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(*, article: str, pn_bar: float, connection: str = "flanged") -> SteelProductDocument:
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
        connection=connection,
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


def test_find_v4_eligible_documents_uses_pn_minimum_semantics() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    eligible = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article=None,
        dn=50,
        pn_bar=16,
        connection=None,
    )

    assert {document.article for document in eligible} == {"a16", "a25", "a40"}
    assert all(document.pn_bar is not None and document.pn_bar >= 16 for document in eligible)


def test_find_v4_eligible_documents_requires_exact_dn_brand_and_connection() -> None:
    documents = [
        _document(article="a25", pn_bar=25, connection="flanged"),
        _document(article="b25", pn_bar=25, connection="welded"),
    ]

    eligible = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article=None,
        dn=50,
        pn_bar=16,
        connection="фланцевое",
    )

    assert [document.article for document in eligible] == ["a25"]


def test_find_v4_eligible_documents_wildcards_missing_pn() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    eligible = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article=None,
        dn=50,
        pn_bar=None,
        connection=None,
    )

    assert {document.article for document in eligible} == {"a10", "a16", "a25", "a40"}


def test_find_v4_eligible_documents_rejects_missing_candidate_pn() -> None:
    documents = [
        SteelProductDocument(
            steel_id="missing",
            article="missing",
            article_norm="missing",
            article_compact="missing",
            name="Temper DN50",
            name_variants=["Temper DN50"],
            brand="Temper",
            dn=50,
            pn_bar=None,
            connection="flanged",
            body_material="сталь 20",
            medium="газ",
            control="ручное",
            temperature="120 c",
            length_mm=None,
            semantic_text="Temper DN50",
            lexical_text="Temper DN50",
            ld_candidates=[],
        )
    ]

    eligible = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article=None,
        dn=50,
        pn_bar=16,
        connection=None,
    )

    assert eligible == []


def test_find_v4_eligible_documents_article_path_is_identity_specific() -> None:
    documents = [
        _document(article="A", pn_bar=25),
        _document(article="B", pn_bar=25),
    ]

    lower_request = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article="A",
        dn=50,
        pn_bar=16,
        connection=None,
    )
    higher_request = find_v4_eligible_documents(
        documents,
        resolved_brand="Temper",
        resolved_article="A",
        dn=50,
        pn_bar=40,
        connection=None,
    )

    assert [document.article for document in lower_request] == ["A"]
    assert higher_request == []


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


def test_build_v4_article_only_normalized_preserves_raw_query_article() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records, _meta = build_v4_cases(documents, target_count=160)
    normalized_case = next(
        record for record in records if record.category == "article_only_normalized"
    )

    assert normalized_case.expected_attributes.article == normalized_case.query
    assert normalized_case.expected_attributes.resolved_article is not None
    assert (
        normalize_article(normalized_case.expected_attributes.article).article_compact
        == normalize_article(normalized_case.expected_attributes.resolved_article).article_compact
    )


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


def test_build_v4_brand_typo_skips_exact_only_brands() -> None:
    documents = [
        SteelProductDocument(
            steel_id="adl-1",
            article="adl-1",
            article_norm="adl1",
            article_compact="adl1",
            name="ADL DN50 PN16",
            name_variants=["ADL DN50 PN16"],
            brand="ADL",
            dn=50,
            pn_bar=16,
            connection="фланцевое",
            body_material="сталь 20",
            medium="вода",
            control="ручное",
            temperature="120 c",
            length_mm=None,
            semantic_text="ADL DN50 PN16",
            lexical_text="ADL DN50 PN16",
            ld_candidates=[],
        ),
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records, meta = build_v4_cases(documents, target_count=160)

    assert all(
        not (
            record.category == "brand_typo"
            and (record.expected_attributes.resolved_brand or "").upper() == "ADL"
        )
        for record in records
    )
    assert meta["adl_available"] is True


def test_build_v4_article_natural_language_uses_brand_and_article_resolution() -> None:
    documents = [
        _document(article="a10", pn_bar=10),
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records, _meta = build_v4_cases(documents, target_count=160)
    natural_language_cases = [
        record for record in records if record.category == "article_natural_language"
    ]

    assert natural_language_cases
    assert all(case.expected_attributes.raw_brand is not None for case in natural_language_cases)
    assert all(case.expected_attributes.article is not None for case in natural_language_cases)
    assert all(
        case.expected_resolution_mode == "brand_and_article"
        for case in natural_language_cases
    )


def test_v3_regression_recomputes_v4_eligible_candidates() -> None:
    documents = [
        _document(article="a16", pn_bar=16),
        _document(article="a25", pn_bar=25),
        _document(article="a40", pn_bar=40),
    ]

    records = _regression_cases(documents, target_count=60)
    pn_case = next(
        record
        for record in records
        if record.expected_attributes.pn_bar == 16 and record.expected_status == "exact_match"
    )

    assert set(pn_case.eligible_competitor_articles) == {"a16", "a25", "a40"}
    assert set(pn_case.preferred_competitor_articles) == {"a16", "a25", "a40"}
