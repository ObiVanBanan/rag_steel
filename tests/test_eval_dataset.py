from __future__ import annotations

import json
from pathlib import Path

ALLOWED_CATEGORIES = {
    "exact_article",
    "modified_article",
    "partial_article",
    "full_name",
    "brand_dn_pn",
    "natural_language",
    "mixed",
    "no_match",
}


def test_eval_dataset_has_expected_shape() -> None:
    dataset_path = Path("eval/queries.jsonl")
    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) >= 500
    assert {record["category"] for record in records} == ALLOWED_CATEGORIES
    assert all(set(record) >= {"query", "category", "expected_ld_articles"} for record in records)
    assert all(isinstance(record["expected_ld_articles"], list) for record in records)
    assert all(
        len(record["expected_ld_articles"]) == len(set(record["expected_ld_articles"]))
        for record in records
    )
    assert all(
        record["expected_ld_articles"] for record in records if record["category"] != "no_match"
    )
    assert any(
        record["category"] == "no_match" and not record["expected_ld_articles"]
        for record in records
    )
