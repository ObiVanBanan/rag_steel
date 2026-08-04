import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_builder import REQUIRED_COLUMNS, profile_csv


def _make_valid_frame() -> pd.DataFrame:
    base = {
        column: [f"{column}-1", f"{column}-2", f"{column}-2"]
        for column in REQUIRED_COLUMNS
    }
    base["steel_article"] = ["A1", "A1", "A1"]
    base["ld_article"] = ["L1", "L2", "L2"]
    base["match_score"] = [7, 8, 8]
    base["match_max"] = [7, 7, 7]
    base["price_ld"] = [100, 200, 200]
    return pd.DataFrame(base)


def test_profile_csv_reports_expected_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    df = _make_valid_frame()
    df.to_csv(csv_path, index=False)

    profile = profile_csv(csv_path)

    assert profile.rows == 3
    assert profile.columns == len(REQUIRED_COLUMNS)
    assert profile.full_duplicates == 1
    assert profile.unique_steel_articles == 1
    assert profile.unique_ld_articles == 2
    assert profile.unique_steel_ld_pairs == 2
    assert profile.null_counts["steel_article"] == 0
    assert profile.match_score_distribution == {"7": 1, "8": 2}
    assert profile.match_max_distribution == {"7": 3}
    assert profile.conflicting_steel_articles_count == 1
    assert profile.conflicting_steel_articles_examples[0]["steel_article"] == "A1"


def test_profile_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"steel_article": ["A1"], "ld_article": ["L1"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        profile_csv(csv_path)


def test_profile_csv_smoke_on_real_dataset() -> None:
    profile = profile_csv(Path("mapping_results.csv"))

    assert profile.rows == 55539
    assert profile.columns == 23
    assert profile.full_duplicates == 11603
    assert profile.unique_steel_articles == 15708
    assert profile.unique_ld_articles == 3280
    assert profile.unique_steel_ld_pairs == 43719
    assert profile.match_max_distribution == {"7": 55539}
    assert set(profile.match_score_distribution) == {"7", "8", "9"}
    assert profile.conflicting_steel_articles_count == 10227
