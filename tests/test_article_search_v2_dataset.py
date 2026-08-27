from __future__ import annotations

import csv
import importlib
import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from eval.build_article_search_v2_dataset import (
    MANDATORY_ANCHORS,
    QUERY_TEMPLATES,
    _first_text,
    _load_source_products,
    _product_family,
    _select_diverse_products,
    _source_products_from_frame,
    build_article_search_v2_dataset,
    build_article_search_v2_records,
)
from rag_steel.schemas import LDProduct, SteelProductDocument


def _document(
    *,
    article: str,
    family: str,
    index: int,
    brand: str | None = None,
) -> SteelProductDocument:
    if family == "butterfly_valve":
        name = f"Затвор дисковый поворотный PALUR {article}"
        body_material = "сталь 20"
        default_brand = "PALUR"
    elif family == "brass_ball_valve":
        name = f"Кран шаровой латунный VALTEC {article}"
        body_material = "латунь"
        default_brand = "Valtec"
    else:
        name = f"Кран шаровой стальной Temper {article}"
        body_material = "сталь 20"
        default_brand = "Temper"

    return SteelProductDocument(
        steel_id=f"{family}-{index}",
        article=article,
        article_norm=article.casefold(),
        article_compact=article.casefold().replace("-", "").replace(".", ""),
        name=name,
        name_variants=[name],
        brand=brand or default_brand,
        dn=15 + index,
        pn_bar=16 + (index % 3) * 9,
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


def _product(
    *,
    article: str,
    family: str,
    brand: str,
    dn: float,
    pn_bar: float,
    pattern_key: str,
    ld_article: str | None = None,
) -> dict[str, object]:
    return {
        "article": article,
        "article_norm": article.casefold(),
        "article_key": article.casefold().replace(".", "").replace("-", ""),
        "brand_key": brand.casefold(),
        "identity_key": f"{article.casefold()}::{brand.casefold()}",
        "family": family,
        "source_brand": brand,
        "dn": dn,
        "pn_bar": pn_bar,
        "connection": "фланцевое",
        "ld_articles": [ld_article or f"LD-{article}"],
        "pattern_key": pattern_key,
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
    *,
    delimiter: str = ",",
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=Path.cwd())


def test_builder_imports_cleanly() -> None:
    module = importlib.import_module("eval.build_article_search_v2_dataset")
    assert module.DEFAULT_OUTPUT_PATH.name == "article_search_v2.jsonl"


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


def test_first_text_skips_none_and_pandas_nan() -> None:
    assert _first_text([None, float("nan"), pd.NA, "  ", "value"]) == "value"
    assert _first_text([None, math.nan, pd.NA]) is None


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
    ball_records = [record for record in records if record["category"] == "ball_valve"]
    assert len(ball_records) == 20
    assert [record["query_variant"] for record in ball_records[:4]] == [
        "need_analog",
        "pick_analog",
        "article_only",
        "replacement_question",
    ]
    assert ball_records[0]["query"].startswith("Нужен аналог BALL-")
    assert ball_records[0]["expected_ld_articles"] == [f"LD-{ball_records[0]['article']}"]
    assert ball_records[0]["expected_source_article_norm"] == ball_records[0]["article"].casefold()
    assert (
        ball_records[0]["expected_source_article_compact"]
        == ball_records[0]["article"].casefold().replace("-", "")
    )
    assert set(meta["query_variants"]) == {name for name, _ in QUERY_TEMPLATES}


def test_build_article_search_v2_records_validates_mandatory_anchor_presence() -> None:
    documents = [_document(article="BALL-01", family="ball_valve", index=1)]

    with pytest.raises(
        ValueError,
        match="Mandatory article_search_v2 anchors are missing from the source bundle",
    ):
        build_article_search_v2_records(
            documents,
            per_family=1,
            mandatory_anchors=("MISSING-ANCHOR",),
        )


def test_canonical_source_alone_uses_adapter_and_populates_source_files() -> None:
    with _workspace_tempdir() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "canonical.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "ld_name": "LD Valve",
                    "ld_article": "LD-1",
                    "ld_url": "https://example.test/ld-1",
                    "ld_dn": 15,
                    "ld_pn_mpa": 1.6,
                    "ld_connection": "фланцевое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Кран шаровой стальной STI 123",
                    "steel_article": "STI-123",
                    "steel_url": "https://example.test/steel-123",
                    "steel_dn": 15,
                    "steel_pn_bar": 16,
                    "steel_connection": "фланцевое",
                    "steel_body_material": "сталь 20",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 1000,
                    "steel_brand": "STI",
                }
            ],
            fieldnames=[
                "ld_name",
                "ld_article",
                "ld_url",
                "ld_dn",
                "ld_pn_mpa",
                "ld_connection",
                "ld_medium",
                "ld_control",
                "ld_temp",
                "ld_length",
                "steel_name",
                "steel_article",
                "steel_url",
                "steel_dn",
                "steel_pn_bar",
                "steel_connection",
                "steel_body_material",
                "steel_medium",
                "steel_control",
                "steel_temp",
                "steel_length",
                "match_score",
                "match_max",
                "price_ld",
                "steel_brand",
            ],
        )

        products, source_files = _load_source_products((csv_path,))

        assert len(products) == 1
        assert products[0]["article"] == "STI-123"
        assert products[0]["source_brand"] == "STI"
        assert products[0]["ld_articles"] == ["LD-1"]
        assert source_files[0]["adapter"] == "canonical"
        assert source_files[0]["name"] == "canonical.csv"


def test_butterfly_source_alone_uses_adapter() -> None:
    with _workspace_tempdir() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "butterfly.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "ld_marking": "LD-BUTTER-1",
                    "ld_dn": 200,
                    "ld_pn": 1.6,
                    "ld_connection": "фланцевое",
                    "ld_control": "ручное",
                    "ld_material": "сталь",
                    "ld_temp": "-10...120",
                    "steel_marking": "PALUR-200",
                    "steel_dn": 200,
                    "steel_pn": 16,
                    "steel_connection": "фланцевое",
                    "steel_control": "ручное",
                    "steel_material": "сталь 20",
                    "steel_temp": "-10...120",
                    "match_score": 8,
                    "match_max": 8,
                }
            ],
            fieldnames=[
                "ld_marking",
                "ld_dn",
                "ld_pn",
                "ld_connection",
                "ld_control",
                "ld_material",
                "ld_temp",
                "steel_marking",
                "steel_dn",
                "steel_pn",
                "steel_connection",
                "steel_control",
                "steel_material",
                "steel_temp",
                "match_score",
                "match_max",
            ],
            delimiter=";",
        )

        products, source_files = _load_source_products((csv_path,))

        assert len(products) == 1
        assert products[0]["family"] == "butterfly_valve"
        assert products[0]["source_brand"] == "PALUR"
        assert source_files[0]["adapter"] == "butterfly"


def test_competitor_source_alone_uses_adapter() -> None:
    with _workspace_tempdir() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "competitor.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "competitor_article": "COMP-15",
                    "ld_article": "LD-COMP-15",
                    "c_brand": "VALTEC",
                    "c_name": "Кран шаровой VALTEC 15",
                    "c_dn": 15,
                    "c_pn": 16,
                    "c_thread_type": "резьба",
                    "c_type": "латунный шаровой",
                    "name": "LD valve",
                    "dn": 15,
                    "pn": 16,
                    "thread_type": "резьба",
                    "type": "кран",
                    "price": 1200,
                    "url": "https://example.test/ld-comp-15",
                }
            ],
            fieldnames=[
                "competitor_article",
                "ld_article",
                "c_brand",
                "c_name",
                "c_dn",
                "c_pn",
                "c_thread_type",
                "c_type",
                "name",
                "dn",
                "pn",
                "thread_type",
                "type",
                "price",
                "url",
            ],
        )

        products, source_files = _load_source_products((csv_path,))

        assert len(products) == 1
        assert products[0]["family"] == "brass_ball_valve"
        assert products[0]["source_brand"] == "Valtec"
        assert source_files[0]["adapter"] == "competitor_ld"


def test_nan_values_are_not_serialized_as_nan() -> None:
    frame = pd.DataFrame(
        [
            {
                "steel_article": "GOOD-1",
                "steel_brand": "STI",
                "steel_name": "Кран шаровой GOOD-1",
                "steel_body_material": "сталь 20",
                "steel_dn": 15,
                "steel_pn_bar": 16,
                "steel_connection": float("nan"),
                "ld_article": "LD-GOOD-1",
            },
            {
                "steel_article": float("nan"),
                "steel_brand": "STI",
                "steel_name": "bad",
                "steel_body_material": "сталь 20",
                "steel_dn": 15,
                "steel_pn_bar": 16,
                "steel_connection": "фланцевое",
                "ld_article": "LD-BAD",
            },
        ]
    )

    products = _source_products_from_frame(frame)

    assert [product["article"] for product in products] == ["GOOD-1"]
    serialized = json.dumps(products, ensure_ascii=False)
    assert '"nan"' not in serialized.casefold()


def test_same_article_under_two_brands_is_excluded_from_exact_match() -> None:
    frame = pd.DataFrame(
        [
            {
                "steel_article": "DUP-1",
                "steel_brand": "BrandA",
                "steel_name": "Кран шаровой BrandA DUP-1",
                "steel_body_material": "сталь 20",
                "steel_dn": 15,
                "steel_pn_bar": 16,
                "steel_connection": "фланцевое",
                "ld_article": "LD-A",
            },
            {
                "steel_article": "dup 1",
                "steel_brand": "BrandB",
                "steel_name": "Кран шаровой BrandB DUP-1",
                "steel_body_material": "сталь 20",
                "steel_dn": 15,
                "steel_pn_bar": 16,
                "steel_connection": "фланцевое",
                "ld_article": "LD-B",
            },
            {
                "steel_article": "UNIQ-1",
                "steel_brand": "BrandA",
                "steel_name": "Кран шаровой BrandA UNIQ-1",
                "steel_body_material": "сталь 20",
                "steel_dn": 20,
                "steel_pn_bar": 25,
                "steel_connection": "фланцевое",
                "ld_article": "LD-U",
            },
        ]
    )

    products = _source_products_from_frame(frame)

    assert [product["article"] for product in products] == ["UNIQ-1"]


def test_diversity_selection_is_deterministic_and_not_all_near_identical() -> None:
    products = [
        _product(
            article=f"PALUR-1000-{index:02d}",
            family="butterfly_valve",
            brand="PALUR",
            dn=1000,
            pn_bar=16,
            pattern_key="alpha:palur",
        )
        for index in range(25)
    ]
    products.extend(
        [
            _product(
                article=f"ABRA-{index:02d}",
                family="butterfly_valve",
                brand="ABRA",
                dn=200 + index * 50,
                pn_bar=10 + (index % 3) * 6,
                pattern_key=f"alpha:abra{index % 4}",
            )
            for index in range(10)
        ]
    )
    products.extend(
        [
            _product(
                article=f"DN{index:03d}-DISC",
                family="butterfly_valve",
                brand="GRANLOCK",
                dn=150 + index * 25,
                pn_bar=25,
                pattern_key=f"mixed:dn{index % 5}",
            )
            for index in range(10)
        ]
    )

    first = _select_diverse_products(products, count=20)
    second = _select_diverse_products(products, count=20)

    assert [product["article"] for product in first] == [product["article"] for product in second]
    assert len({product["source_brand"] for product in first}) >= 3
    assert len({product["dn"] for product in first}) >= 8
    assert len({product["pattern_key"] for product in first}) >= 5
    assert sum(1 for product in first if product["source_brand"] == "PALUR") < 20


def test_mandatory_anchor_is_included_when_present() -> None:
    with _workspace_tempdir() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "canonical.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "ld_name": "LD Anchor",
                    "ld_article": "LD-ANCHOR",
                    "ld_url": None,
                    "ld_dn": 15,
                    "ld_pn_mpa": 4.0,
                    "ld_connection": "фланцевое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Кран шаровой anchor",
                    "steel_article": "КШ.ФПР.015.40-01",
                    "steel_url": None,
                    "steel_dn": 15,
                    "steel_pn_bar": 40,
                    "steel_connection": "фланцевое",
                    "steel_body_material": "сталь 20",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 1000,
                    "steel_brand": "ALSO",
                },
                {
                    "ld_name": "LD Anchor",
                    "ld_article": "LD-ANCHOR-2",
                    "ld_url": None,
                    "ld_dn": 15,
                    "ld_pn_mpa": 4.0,
                    "ld_connection": "фланцевое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Кран шаровой anchor",
                    "steel_article": MANDATORY_ANCHORS[0],
                    "steel_url": None,
                    "steel_dn": 15,
                    "steel_pn_bar": 40,
                    "steel_connection": "фланцевое",
                    "steel_body_material": "сталь 20",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 1000,
                    "steel_brand": "ALSO",
                },
                {
                    "ld_name": "LD Brass",
                    "ld_article": "LD-BRASS",
                    "ld_url": None,
                    "ld_dn": 20,
                    "ld_pn_mpa": 1.6,
                    "ld_connection": "резьбовое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Кран шаровой латунный BRASS-1",
                    "steel_article": "BRASS-1",
                    "steel_url": None,
                    "steel_dn": 20,
                    "steel_pn_bar": 16,
                    "steel_connection": "резьбовое",
                    "steel_body_material": "латунь",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 900,
                    "steel_brand": "VALTEC",
                },
                {
                    "ld_name": "LD Butterfly",
                    "ld_article": "LD-BUTTER",
                    "ld_url": None,
                    "ld_dn": 200,
                    "ld_pn_mpa": 1.6,
                    "ld_connection": "фланцевое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Затвор дисковый поворотный PALUR-200",
                    "steel_article": "PALUR-200",
                    "steel_url": None,
                    "steel_dn": 200,
                    "steel_pn_bar": 16,
                    "steel_connection": "фланцевое",
                    "steel_body_material": "сталь 20",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 1900,
                    "steel_brand": "PALUR",
                },
            ],
            fieldnames=[
                "ld_name",
                "ld_article",
                "ld_url",
                "ld_dn",
                "ld_pn_mpa",
                "ld_connection",
                "ld_medium",
                "ld_control",
                "ld_temp",
                "ld_length",
                "steel_name",
                "steel_article",
                "steel_url",
                "steel_dn",
                "steel_pn_bar",
                "steel_connection",
                "steel_body_material",
                "steel_medium",
                "steel_control",
                "steel_temp",
                "steel_length",
                "match_score",
                "match_max",
                "price_ld",
                "steel_brand",
            ],
        )

        records, _ = build_article_search_v2_dataset(
            csv_paths=(csv_path,),
            output_path=tmp / "out.jsonl",
            meta_path=tmp / "out.meta.json",
            per_family=1,
            mandatory_anchors=(MANDATORY_ANCHORS[0],),
        )

        ball_records = [record for record in records if record["category"] == "ball_valve"]
        assert [record["article"] for record in ball_records] == [MANDATORY_ANCHORS[0]]
        assert ball_records[0]["expected_ld_articles"] == ["LD-ANCHOR", "LD-ANCHOR-2"]


def test_mandatory_anchor_missing_fails_explicitly() -> None:
    with _workspace_tempdir() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "canonical.csv"
        _write_csv(
            csv_path,
            rows=[
                {
                    "ld_name": "LD Valve",
                    "ld_article": "LD-1",
                    "ld_url": None,
                    "ld_dn": 15,
                    "ld_pn_mpa": 1.6,
                    "ld_connection": "фланцевое",
                    "ld_medium": None,
                    "ld_control": None,
                    "ld_temp": None,
                    "ld_length": None,
                    "steel_name": "Кран шаровой стальной STI 123",
                    "steel_article": "STI-123",
                    "steel_url": None,
                    "steel_dn": 15,
                    "steel_pn_bar": 16,
                    "steel_connection": "фланцевое",
                    "steel_body_material": "сталь 20",
                    "steel_medium": None,
                    "steel_control": None,
                    "steel_temp": None,
                    "steel_length": None,
                    "match_score": 1,
                    "match_max": 1,
                    "price_ld": 1000,
                    "steel_brand": "STI",
                }
            ],
            fieldnames=[
                "ld_name",
                "ld_article",
                "ld_url",
                "ld_dn",
                "ld_pn_mpa",
                "ld_connection",
                "ld_medium",
                "ld_control",
                "ld_temp",
                "ld_length",
                "steel_name",
                "steel_article",
                "steel_url",
                "steel_dn",
                "steel_pn_bar",
                "steel_connection",
                "steel_body_material",
                "steel_medium",
                "steel_control",
                "steel_temp",
                "steel_length",
                "match_score",
                "match_max",
                "price_ld",
                "steel_brand",
            ],
        )

        with pytest.raises(ValueError, match="Mandatory article_search_v2 anchors are missing"):
            build_article_search_v2_dataset(
                csv_paths=(csv_path,),
                output_path=tmp / "out.jsonl",
                meta_path=tmp / "out.meta.json",
                per_family=1,
            )
