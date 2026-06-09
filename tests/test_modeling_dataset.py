from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_modeling_dataset as modeling  # noqa: E402


def row(
    row_id: str,
    state: str,
    district: str,
    crop: str,
    season: str,
    year: int,
    target: float,
) -> dict[str, object]:
    item: dict[str, object] = {
        "canonical_crop_row_id": row_id,
        "district_id": f"DIST_{state}_{district}",
        "canonical_state_name": state,
        "canonical_district_name": district,
        "Crop_canonical": crop,
        "Season_canonical": season,
        "Crop_Year": year,
        "Area_corrected": 10.0,
        "latitude": 12.0,
        "longitude": 77.0,
        "target_yield": target,
        "basic_model_eligibility": 1,
    }
    for column in modeling.WEATHER_FEATURES:
        item[column] = 1.0
    return item


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("r1997", "A", "D1", "Rice", "Kharif", 1997, 1.0),
            row("r1998", "A", "D1", "Rice", "Kharif", 1998, 2.0),
            row("r2000", "A", "D1", "Rice", "Kharif", 2000, 4.0),
            row("other_district_1997", "A", "D2", "Rice", "Kharif", 1997, 10.0),
            row("other_crop_1997", "A", "D1", "Wheat", "Kharif", 1997, 11.0),
            row("other_season_1997", "A", "D1", "Rice", "Rabi", 1997, 12.0),
            row("train2010", "A", "D3", "Maize", "Kharif", 2010, 3.0),
            row("validation2011", "NewState", "NewDistrict", "NewCrop", "Kharif", 2011, 5.0),
            row("validation2012", "A", "D3", "Maize", "Kharif", 2012, 6.0),
            row("test2013", "A", "D4", "Sorghum", "Summer", 2013, 7.0),
            row("test2014", "A", "D4", "Sorghum", "Summer", 2014, 8.0),
        ]
    )


def build_processed() -> pd.DataFrame:
    return modeling.build_processed_dataset(synthetic_frame(), expected_rows=None)


def test_exact_previous_year_creates_lag() -> None:
    processed = build_processed()
    current = processed[processed["canonical_crop_row_id"].eq("r1998")].iloc[0]
    assert current["lag_available"] == 1
    assert current["lag_yield_1y"] == 1.0


def test_two_year_gap_does_not_create_lag() -> None:
    processed = build_processed()
    current = processed[processed["canonical_crop_row_id"].eq("r2000")].iloc[0]
    assert current["lag_available"] == 0
    assert pd.isna(current["lag_yield_1y"])


def test_lag_does_not_come_from_other_district() -> None:
    frame = pd.DataFrame(
        [
            row("d2_1997", "A", "D2", "Rice", "Kharif", 1997, 10.0),
            row("d1_1998", "A", "D1", "Rice", "Kharif", 1998, 2.0),
            row("filler_2014", "A", "D1", "Rice", "Kharif", 2014, 3.0),
        ]
    )
    processed = modeling.build_processed_dataset(frame, expected_rows=None)
    assert processed.loc[processed["canonical_crop_row_id"].eq("d1_1998"), "lag_available"].iloc[0] == 0


def test_lag_does_not_come_from_other_crop() -> None:
    frame = pd.DataFrame(
        [
            row("wheat_1997", "A", "D1", "Wheat", "Kharif", 1997, 10.0),
            row("rice_1998", "A", "D1", "Rice", "Kharif", 1998, 2.0),
            row("filler_2014", "A", "D1", "Rice", "Kharif", 2014, 3.0),
        ]
    )
    processed = modeling.build_processed_dataset(frame, expected_rows=None)
    assert processed.loc[processed["canonical_crop_row_id"].eq("rice_1998"), "lag_available"].iloc[0] == 0


def test_lag_does_not_come_from_other_season() -> None:
    frame = pd.DataFrame(
        [
            row("rabi_1997", "A", "D1", "Rice", "Rabi", 1997, 10.0),
            row("kharif_1998", "A", "D1", "Rice", "Kharif", 1998, 2.0),
            row("filler_2014", "A", "D1", "Rice", "Kharif", 2014, 3.0),
        ]
    )
    processed = modeling.build_processed_dataset(frame, expected_rows=None)
    assert processed.loc[processed["canonical_crop_row_id"].eq("kharif_1998"), "lag_available"].iloc[0] == 0


def test_lag_source_year_is_always_y_minus_one() -> None:
    processed = build_processed()
    available = processed[processed["lag_available"].eq(1)]
    assert (available["lag_source_crop_year"].astype(int) == available["Crop_Year"].astype(int) - 1).all()


def test_split_2010_is_train() -> None:
    assert modeling.split_for_year(2010) == "train"


def test_split_2011_is_validation() -> None:
    assert modeling.split_for_year(2011) == "validation"


def test_split_2012_is_validation() -> None:
    assert modeling.split_for_year(2012) == "validation"


def test_split_2013_is_test() -> None:
    assert modeling.split_for_year(2013) == "test"


def test_split_2014_is_test() -> None:
    assert modeling.split_for_year(2014) == "test"


def test_unsupported_year_raises_error() -> None:
    with pytest.raises(ValueError, match="Unsupported Crop_Year"):
        modeling.split_for_year(2015)


def test_forbidden_leakage_column_is_not_feature() -> None:
    for columns in modeling.feature_sets().values():
        assert "Production_corrected" not in columns
        assert "selected_source" not in columns


def test_target_is_not_feature() -> None:
    for columns in modeling.feature_sets().values():
        assert "target_yield" not in columns


def test_processed_dataset_preserves_row_count() -> None:
    frame = synthetic_frame()
    processed = modeling.build_processed_dataset(frame, expected_rows=None)
    assert len(processed) == len(frame)


def test_processed_dataset_has_unique_row_identifier() -> None:
    processed = build_processed()
    assert processed["canonical_crop_row_id"].is_unique


def test_unseen_category_audit_finds_new_validation_category() -> None:
    processed = build_processed()
    unseen_rows, summary = modeling.audit_unseen_categories(processed)
    assert "NewState" in unseen_rows["unseen_value"].tolist()
    state_summary = summary[summary["categorical_column"].eq("canonical_state_name")].iloc[0]
    assert state_summary["validation_unseen_count"] >= 1


def test_unseen_category_is_not_removed() -> None:
    processed = build_processed()
    assert processed["canonical_state_name"].eq("NewState").any()


def test_feature_set_with_lag_contains_lag() -> None:
    assert "lag_yield_1y" in modeling.feature_sets()["core_with_lag"]
    assert "lag_available" in modeling.feature_sets()["core_with_lag"]


def test_feature_set_without_lag_excludes_lag() -> None:
    assert "lag_yield_1y" not in modeling.feature_sets()["core_without_lag"]
    assert "lag_available" not in modeling.feature_sets()["core_without_lag"]
