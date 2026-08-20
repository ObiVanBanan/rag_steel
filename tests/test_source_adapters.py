from __future__ import annotations

from pathlib import Path

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
