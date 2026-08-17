from __future__ import annotations

from eval.build_v3_eval_dataset import _expected_for_query, build_generated_v3_cases, build_v3_cases
from eval.v3_common import find_hard_eligible_documents, score_preferred_documents
from eval.v3_schema import ExpectedAttributes
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(
    *,
    article: str,
    name: str,
    brand: str = "Temper",
    dn: float = 80,
    pn_bar: float = 16,
    connection: str = "сварное",
    body_material: str | None = "сталь 20",
    medium: str | None = "газ",
    control: str | None = "ручное",
    temperature: str | None = "120 c",
    series: str | None = "60",
) -> SteelProductDocument:
    return SteelProductDocument(
        steel_id=article,
        article=article,
        article_norm=article.casefold(),
        article_compact=article.casefold().replace("-", ""),
        name=name,
        name_variants=[name],
        brand=brand,
        dn=dn,
        pn_bar=pn_bar,
        connection=connection,
        body_material=body_material,
        medium=medium,
        control=control,
        temperature=temperature,
        length_mm=None,
        semantic_text=name,
        lexical_text=name,
        ld_candidates=[
            LDProduct(
                article=f"ld-{article}",
                article_norm=f"ld-{article}".casefold(),
                name=f"LD {name}",
            )
        ],
    )


def test_hard_eligible_filters_only_hard_fields() -> None:
    documents = [
        _document(article="a1", name="Temper A", body_material="сталь 20"),
        _document(article="a2", name="Temper B", body_material="сталь 09г2с"),
        _document(article="b1", name="Broen A", brand="Broen"),
    ]

    eligible = find_hard_eligible_documents(documents, "Temper", 80, 16, "сварное")

    assert [document.article for document in eligible] == ["a1", "a2"]


def test_preferred_scoring_promotes_soft_match_and_falls_back() -> None:
    documents = [
        _document(article="a1", name="Temper A", body_material="сталь 20", medium="газ"),
        _document(article="a2", name="Temper B", body_material="сталь 09г2с", medium="газ"),
    ]

    preferred = score_preferred_documents(
        documents,
        ExpectedAttributes(body_material="сталь 20", medium="газ"),
    )

    assert [document.article for document in preferred] == ["a1"]

    fallback = score_preferred_documents(
        documents,
        ExpectedAttributes(body_material="сталь 404"),
    )

    assert [document.article for document in fallback] == ["a1", "a2"]


def test_build_v3_cases_is_deterministic_and_keeps_required_categories() -> None:
    documents = [
        _document(article="1184399", name="Temper series 14 DN80 PN16 A"),
        _document(
            article="38645080",
            name="Broen DN50 PN40 B",
            brand="Broen",
            dn=50,
            pn_bar=40,
            connection="фланцевое",
            body_material="сталь 09г2с",
            medium="вода",
            control="ручное",
            temperature="90 c",
            series="22",
        ),
    ]

    records_a, meta_a = build_v3_cases(documents, target_count=40)
    records_b, meta_b = build_v3_cases(documents, target_count=40)

    assert [record.model_dump(mode="json") for record in records_a] == [
        record.model_dump(mode="json") for record in records_b
    ]
    assert meta_a == meta_b
    categories = {record.category for record in records_a}
    assert "no_brand" in categories
    assert "impossible_hard" in categories
    assert "hard_plus_material" in categories
    assert "compact_syntax" in categories
    assert "hard_plus_series" in categories


def test_expected_for_query_only_keeps_attributes_present_in_query() -> None:
    document = _document(
        article="1184399",
        name="Temper DN80 PN16",
        body_material="сталь 09г2с",
        medium="газ",
        control="ручное",
        temperature="120 c",
    )

    material = _expected_for_query(document, category="hard_plus_material")
    assert material.body_material == "сталь 09г2с"
    assert material.medium is None
    assert material.control is None
    assert material.temperature is None

    compact = _expected_for_query(document, category="compact_syntax")
    assert compact.connection is None

    alias = _expected_for_query(document, category="russian_alias")
    assert alias.connection is None

    impossible = _expected_for_query(document, category="impossible_hard")
    assert impossible.connection is None
    assert impossible.dn == 999
    assert impossible.pn_bar == 999


def test_generated_v3_cases_include_negative_categories() -> None:
    documents = [
        _document(article=f"t-{idx}", name=f"Temper {idx}", brand="Temper", dn=50 + idx, pn_bar=16)
        for idx in range(1, 4)
    ] + [
        _document(article=f"b-{idx}", name=f"Broen {idx}", brand="Broen", dn=60 + idx, pn_bar=16)
        for idx in range(1, 4)
    ]

    records, meta = build_generated_v3_cases(documents, target_count=24)

    categories = {record.category for record in records}
    assert "no_brand" in categories
    assert "impossible_hard" in categories
    assert meta["by_category"]["no_brand"] >= 1
    assert meta["by_category"]["impossible_hard"] >= 1
