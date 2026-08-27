from __future__ import annotations

from eval.build_article_search_v2_dataset import (
    QUERY_TEMPLATES,
    _product_family,
    build_article_search_v2_records,
)
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(
    *,
    article: str,
    family: str,
    index: int,
) -> SteelProductDocument:
    if family == "butterfly_valve":
        name = f"Затвор дисковый поворотный PALUR {article}"
        body_material = "сталь 20"
        brand = "PALUR"
    elif family == "brass_ball_valve":
        name = f"Кран шаровой латунный VALTEC {article}"
        body_material = "латунь"
        brand = "Valtec"
    else:
        name = f"Кран шаровой стальной Temper {article}"
        body_material = "сталь 20"
        brand = "Temper"

    return SteelProductDocument(
        steel_id=f"{family}-{index}",
        article=article,
        article_norm=article.casefold(),
        article_compact=article.casefold().replace("-", "").replace(".", ""),
        name=name,
        name_variants=[name],
        brand=brand,
        dn=15 + index,
        pn_bar=16,
        connection="фланцевое",
        body_material=body_material,
        medium=None,
        control=None,
        temperature=None,
        length_mm=None,
        url=None,
        semantic_text="",
        lexical_text="",
        ld_candidates=[
            LDProduct(
                article=f"LD-{article}",
                article_norm=f"ld-{article}".casefold(),
                name=f"LD {article}",
                url=None,
                dn=15 + index,
                pn_bar=16,
                connection="фланцевое",
            )
        ],
    )


def test_product_family_detects_ball_brass_and_butterfly() -> None:
    assert _product_family(_document(article="A-1", family="ball_valve", index=1)) == "ball_valve"
    assert (
        _product_family(_document(article="B-1", family="brass_ball_valve", index=1))
        == "brass_ball_valve"
    )
    assert (
        _product_family(_document(article="C-1", family="butterfly_valve", index=1))
        == "butterfly_valve"
    )


def test_build_article_search_v2_records_produces_balanced_families() -> None:
    documents = []
    for family, prefix in (
        ("ball_valve", "BALL"),
        ("brass_ball_valve", "BRASS"),
        ("butterfly_valve", "BUTTER"),
    ):
        for index in range(20):
            documents.append(_document(article=f"{prefix}-{index:02d}", family=family, index=index))

    records, meta = build_article_search_v2_records(documents, per_family=20)

    assert len(records) == 60
    assert meta["per_family"] == {
        "ball_valve": 20,
        "brass_ball_valve": 20,
        "butterfly_valve": 20,
    }
    assert records[0]["query"] == "Нужен аналог BALL-00"
    assert records[1]["query"] == "Подбери аналог BALL-01"
    assert records[2]["query"] == "BALL-02"
    assert records[3]["query"] == "Что есть вместо BALL-03?"
    assert records[0]["expected_ld_articles"] == ["LD-BALL-00"]
    assert set(meta["query_variants"]) == {name for name, _ in QUERY_TEMPLATES}
