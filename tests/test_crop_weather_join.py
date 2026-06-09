from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_crop_weather_dataset as joiner  # noqa: E402


def crop_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "State_Name": ["A", "A"],
            "District_Name": ["D", "D"],
            "Crop_Year": [1997, 1998],
            "Season": ["Kharif", "Kharif"],
            "Crop": ["Rice", "Rice"],
            "source_row_id": [10, 11],
            "start_date": ["1997-06-01", "1998-06-01"],
            "end_date": ["1997-06-10", "1998-06-10"],
            "district_id": ["DIST", "DIST"],
            "weather_point_id": ["WPT", "WPT"],
            "weather_window_id": ["WW1", "WW2"],
            "point_assignment_confidence": ["confirmed", "confirmed"],
        }
    )


def feature_row(window_id: str, start_date: str, end_date: str) -> dict[str, object]:
    row: dict[str, object] = {
        "weather_window_id": window_id,
        "weather_point_id": "WPT",
        "start_date": start_date,
        "end_date": end_date,
    }
    for column in joiner.FEATURE_COLUMNS:
        row[column] = True if column == "weather_window_valid" else 1.0
    row["weather_days_expected"] = 10
    row["weather_days_present"] = 10
    row["weather_coverage_ratio"] = 1.0
    return row


def features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            feature_row("WW1", "1997-06-01", "1997-06-10"),
            feature_row("WW2", "1998-06-01", "1998-06-10"),
        ]
    )


def test_many_to_one_join_preserves_crop_row_count() -> None:
    crop = crop_frame()
    features = features_frame()
    joined = joiner.build_joined_dataset(crop, features)
    assert len(joined) == len(crop)


def test_duplicate_weather_window_id_in_features_raises_error() -> None:
    crop = crop_frame()
    features = pd.concat([features_frame(), features_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="weather_window_id must be unique"):
        joiner.build_joined_dataset(crop, features)


def test_missing_weather_window_raises_error() -> None:
    crop = crop_frame()
    features = features_frame().loc[lambda frame: frame["weather_window_id"] != "WW2"]
    with pytest.raises(ValueError, match="Missing weather feature rows"):
        joiner.build_joined_dataset(crop, features)


def test_join_preserves_source_row_id() -> None:
    crop = crop_frame()
    joined = joiner.build_joined_dataset(crop, features_frame())
    assert joined["source_row_id"].tolist() == [10, 11]


def test_all_weather_features_are_attached() -> None:
    joined = joiner.build_joined_dataset(crop_frame(), features_frame())
    assert set(joiner.FEATURE_COLUMNS).issubset(joined.columns)
    assert int(joined[joiner.FEATURE_COLUMNS].isna().sum().sum()) == 0
