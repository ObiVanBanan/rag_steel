from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_steel.source_adapters import (
    combined_source_sha256,
    detect_source_adapter,
    load_source_bundle,
    load_source_frame,
)


def test_detect_and_load_real_source_adapters() -> None:
    canonical_path = Path("mapping_results.csv")
    butterfly_path = Path("butterfly_mapping_results.csv")
    competitor_path = Path("competitor_ld_mapping.csv")

    canonical_frame, canonical_record = load_source_frame(canonical_path)
    butterfly_frame, butterfly_record = load_source_frame(butterfly_path)
    competitor_frame, competitor_record = load_source_frame(competitor_path)

    assert detect_source_adapter(canonical_path) == "canonical"
    assert detect_source_adapter(butterfly_path) == "butterfly"
    assert detect_source_adapter(competitor_path) == "competitor_ld"

    assert len(canonical_frame) == 55539
    assert len(butterfly_frame) == 11648
    assert len(competitor_frame) == 3679

    assert canonical_record.adapter == "canonical"
    assert butterfly_record.adapter == "butterfly"
    assert competitor_record.adapter == "competitor_ld"

    assert set(butterfly_frame["steel_brand"].dropna().unique()) == {"PALUR"}
    assert set(competitor_frame["steel_brand"].dropna().unique()) == {
        "Aquasfera",
        "Gallop",
        "STI",
        "Stout",
        "Valtec",
        "БАЗ",
    }


def test_load_source_bundle_combines_multiple_sources() -> None:
    source_paths = [
        Path("mapping_results.csv"),
        Path("butterfly_mapping_results.csv"),
        Path("competitor_ld_mapping.csv"),
    ]

    frame, source_files = load_source_bundle(source_paths)

    assert len(frame) == 70866
    assert [record.name for record in source_files] == [
        "mapping_results.csv",
        "butterfly_mapping_results.csv",
        "competitor_ld_mapping.csv",
    ]
    assert [record.adapter for record in source_files] == [
        "canonical",
        "butterfly",
        "competitor_ld",
    ]
    assert combined_source_sha256(source_files)


def test_butterfly_adapter_maps_representative_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "butterfly.csv"
    pd.DataFrame(
        [
            {
                "ld_marking": "LD-123",
                "ld_dn": 50,
                "ld_pn": 16,
                "ld_connection": "Фланцевое",
                "ld_control": "Ручное",
                "ld_material": "Сталь 20",
                "ld_temp": "-40..200",
                "steel_marking": "PALUR-123",
                "steel_dn": 50,
                "steel_pn": 16,
                "steel_connection": "Фланцевое",
                "steel_control": "Ручное",
                "steel_material": "Сталь 20",
                "steel_temp": "-40..400",
                "match_score": 6,
                "match_max": 6,
            }
        ]
    ).to_csv(csv_path, index=False, sep=";")

    frame, record = load_source_frame(csv_path)

    assert record.adapter == "butterfly"
    assert frame.iloc[0]["ld_article"] == "LD-123"
    assert frame.iloc[0]["ld_name"] == "Затвор дисковый поворотный LD-123"
    assert frame.iloc[0]["steel_article"] == "PALUR-123"
    assert frame.iloc[0]["steel_name"] == "Затвор дисковый поворотный PALUR PALUR-123"
    assert frame.iloc[0]["steel_brand"] == "PALUR"
    assert frame.iloc[0]["ld_url"] is None
    assert frame.iloc[0]["steel_url"] is None


def test_competitor_adapter_maps_url_and_material_from_competitor_side(tmp_path: Path) -> None:
    csv_path = tmp_path / "competitor.csv"
    pd.DataFrame(
        [
            {
                "competitor_article": "SVB-1",
                "ld_article": "LD-1",
                "c_brand": "Stout",
                "c_name": "STOUT кран шаровой",
                "c_dn": 20,
                "c_pn": 40,
                "c_thread_type": "Внутренняя/Наружная",
                "c_type": "кран угловой",
                "name": "LD Pride латунный",
                "dn": 20,
                "pn": 40,
                "thread_type": "Внутренняя/Наружная",
                "type": "кран латунный",
                "price": 587,
                "url": "https://example.test/ld",
            }
        ]
    ).to_csv(csv_path, index=False)

    frame, record = load_source_frame(csv_path)

    assert record.adapter == "competitor_ld"
    assert frame.iloc[0]["ld_article"] == "LD-1"
    assert frame.iloc[0]["steel_article"] == "SVB-1"
    assert frame.iloc[0]["steel_brand"] == "Stout"
    assert frame.iloc[0]["ld_url"] == "https://example.test/ld"
    assert frame.iloc[0]["steel_url"] is None
    assert frame.iloc[0]["steel_body_material"] is None


def test_competitor_adapter_infers_brass_from_competitor_fields_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "competitor_brass.csv"
    pd.DataFrame(
        [
            {
                "competitor_article": "SVB-2",
                "ld_article": "LD-2",
                "c_brand": "Valtec",
                "c_name": "VALTEC кран латунный",
                "c_dn": 20,
                "c_pn": 40,
                "c_thread_type": "Внутренняя/Наружная",
                "c_type": "кран латунный",
                "name": "LD Pride",
                "dn": 20,
                "pn": 40,
                "thread_type": "Внутренняя/Наружная",
                "type": "кран стальной",
                "price": 587,
                "url": "https://example.test/ld2",
            }
        ]
    ).to_csv(csv_path, index=False)

    frame, _ = load_source_frame(csv_path)

    assert frame.iloc[0]["steel_body_material"] == "латунь"
