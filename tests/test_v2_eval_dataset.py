from __future__ import annotations

from eval.build_v2_eval_dataset import (
    _find_eligible_documents,
    _is_mojibake_text,
    build_v2_cases,
)
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(
    *,
    article: str,
    name: str,
    brand: str = "Temper",
    dn: float = 80,
    pn_bar: float = 16,
    connection: str | None = "фланцевое",
    body_material: str | None = "сталь 20",
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


def test_one_constraint_can_match_multiple_competitors() -> None:
    documents = [
        _document(article="1184399", name="Temper DN80 PN16 A"),
        _document(article="38645080", name="Temper DN80 PN16 B"),
        _document(article="999", name="Broen DN80 PN16", brand="Broen"),
    ]

    eligible = _find_eligible_documents(
        documents,
        {
            "brand": "Temper",
            "dn": 80,
            "pn_bar": 16,
            "connection": None,
            "series": None,
            "body_material": None,
        },
    )

    assert [document.article for document in eligible] == ["1184399", "38645080"]


def test_material_constraint_narrows_eligible_set() -> None:
    documents = [
        _document(article="1184399", name="Temper DN80 PN16 A", body_material="сталь 20"),
        _document(article="38645080", name="Temper DN80 PN16 B", body_material="сталь 09г2с"),
    ]

    eligible = _find_eligible_documents(
        documents,
        {
            "brand": "Temper",
            "dn": 80,
            "pn_bar": 16,
            "connection": None,
            "series": None,
            "body_material": "сталь 20",
        },
    )

    assert [document.article for document in eligible] == ["1184399"]


def test_wrong_pn_produces_empty_gold() -> None:
    documents = [
        _document(article="1184399", name="Temper DN80 PN16 A"),
        _document(article="38645080", name="Temper DN80 PN16 B"),
    ]

    eligible = _find_eligible_documents(
        documents,
        {
            "brand": "Temper",
            "dn": 80,
            "pn_bar": 999,
            "connection": None,
            "series": None,
            "body_material": None,
        },
    )

    assert eligible == []


def test_wrong_material_is_not_created_if_combination_exists() -> None:
    documents = [
        _document(article="1184399", name="Temper DN80 PN16 A", body_material="сталь 20"),
        _document(article="38645080", name="Temper DN80 PN16 B", body_material="сталь 09г2с"),
        _document(
            article="38645081",
            name="Temper DN80 PN16 C",
            body_material="нержавеющая сталь",
        ),
    ]

    records, meta = build_v2_cases(documents, document_limit=2)

    assert meta["positive"] > 0
    assert not any(record["category"] == "wrong_material" for record in records)


def test_article_and_name_gold_and_duplicate_queries_are_deterministic() -> None:
    documents = [
        _document(article="A-0486", name="Valve Exact Name"),
        _document(article="1184399", name="Temper DN80 PN16"),
        _document(article="1184398", name="Temper DN80 PN16"),
    ]

    records_a, meta_a = build_v2_cases(documents, document_limit=3)
    records_b, meta_b = build_v2_cases(documents, document_limit=3)

    assert records_a == records_b
    assert meta_a == meta_b
    assert any(
        record["category"] == "article_exact" and record["query"] == "A-0486"
        for record in records_a
    )
    assert any(
        record["category"] == "name_exact"
        and record["query"] == "Valve Exact Name"
        for record in records_a
    )
    normalized_keys = [" ".join(record["query"].split()).casefold() for record in records_a]
    assert len(normalized_keys) == len(set(normalized_keys))


def test_normal_russian_text_is_not_mojibake() -> None:
    assert _is_mojibake_text("РЎС‚Р°Р»СЊ 20") is False
    assert _is_mojibake_text("РЎРІР°СЂРЅРѕР№ РєСЂР°РЅ") is False
    assert _is_mojibake_text("Р РµР·СЊР±РѕРІРѕР№ РєСЂР°РЅ") is False
    assert _is_mojibake_text("Р Сѓ16") is False


def test_mojibake_detector_matches_known_markers() -> None:
    assert _is_mojibake_text("Р В°0486") is True
    assert _is_mojibake_text("Broen Р вЂќРЎС“80 Р  РЎС“16") is True
    assert _is_mojibake_text("РЎвЂћР В»Р В°Р Р…РЎвЂ Р ВµР Р†РЎвЂ№Р в„–") is True


def test_natural_language_case_uses_full_expected_constraints() -> None:
    documents = [
        _document(
            article="1184399",
            name="Temper DN80 PN16 A",
            connection="фланцевое",
            body_material="сталь 20",
        ),
        _document(
            article="38645080",
            name="Temper DN80 PN16 B",
            connection="сварное",
            body_material="сталь 20",
        ),
        _document(
            article="38645081",
            name="Temper DN80 PN16 C",
            connection="фланцевое",
            body_material="сталь 09г2с",
        ),
    ]

    records, _ = build_v2_cases(documents, document_limit=3)
    natural_case = next(record for record in records if record["category"] == "natural_language")

    assert natural_case["expected_constraints"]["brand"] == "Temper"
    assert natural_case["expected_constraints"]["dn"] == 80
    assert natural_case["expected_constraints"]["pn_bar"] == 16
    assert natural_case["expected_constraints"]["connection"] == "фланцевое"
    assert natural_case["expected_constraints"]["body_material"] == "сталь 20"
    assert natural_case["eligible_competitor_articles"] == ["1184399"]


def test_wrong_known_brand_uses_only_parser_supported_brands() -> None:
    documents = [
        _document(article="1184399", name="Temper DN80 PN16 A", brand="Temper"),
        _document(article="38645080", name="Danfoss DN80 PN16 B", brand="Danfoss"),
        _document(article="38645081", name="Broen DN80 PN16 C", brand="Broen"),
    ]

    records, _ = build_v2_cases(documents, document_limit=3)
    wrong_known_brand = [
        record for record in records if record["category"] == "wrong_known_brand"
    ]

    assert all(
        record["expected_constraints"]["brand"]
        in {"Temper", "Broen", "ALSO", "MARSHAL", "Бивал", "ADL", "FORTECA"}
        for record in wrong_known_brand
    )
