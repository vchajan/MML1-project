from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import aggregate_crop_weather_windows as agg  # noqa: E402


def weather_frame(dates: list[str], precipitation: list[float | None] | None = None) -> pd.DataFrame:
    precipitation = precipitation if precipitation is not None else [1.0] * len(dates)
    return pd.DataFrame(
        {
            "weather_point_id": ["WPT"] * len(dates),
            "date": dates,
            "PRECTOTCORR": precipitation,
            "T2M": [20.0 + index for index in range(len(dates))],
            "T2M_MAX": [30.0 + index for index in range(len(dates))],
            "T2M_MIN": [12.0 - index for index in range(len(dates))],
            "RH2M": [60.0] * len(dates),
            "ALLSKY_SFC_SW_DWN": [15.0] * len(dates),
            "WS2M": [2.0] * len(dates),
        }
    )


def test_date_interval_is_inclusive() -> None:
    result = agg.aggregate_weather_window(weather_frame(["2001-01-01", "2001-01-02", "2001-01-03"]), "2001-01-02", "2001-01-03")
    assert result["weather_days_expected"] == 2
    assert result["weather_days_present"] == 2


def test_leap_day_is_counted() -> None:
    result = agg.aggregate_weather_window(weather_frame(["2020-02-28", "2020-02-29", "2020-03-01"]), "2020-02-28", "2020-03-01")
    assert result["weather_days_expected"] == 3
    assert result["weather_days_present"] == 3


def test_longest_dry_spell_works() -> None:
    precipitation = pd.Series([0.0, 0.2, None, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert agg.calculate_longest_dry_spell(precipitation) == 3


def test_longest_wet_spell_works() -> None:
    precipitation = pd.Series([1.0, 2.0, None, 5.0, 0.0, 1.0, 1.0, 1.0])
    assert agg.calculate_longest_wet_spell(precipitation) == 3


def test_missing_precipitation_is_not_dry_or_wet() -> None:
    result = agg.aggregate_weather_window(weather_frame(["2001-01-01", "2001-01-02", "2001-01-03"], [None, 0.0, 3.0]), "2001-01-01", "2001-01-03")
    assert result["dry_days_lt1mm"] == 1
    assert result["rainy_days_ge1mm"] == 1
    assert result["longest_dry_spell_days"] == 1
    assert result["longest_wet_spell_days"] == 1


def test_weather_event_counts_are_correct() -> None:
    frame = weather_frame(["2001-01-01", "2001-01-02", "2001-01-03"], [0.0, 1.0, 25.0])
    frame["T2M_MAX"] = [34.0, 35.0, 36.0]
    frame["T2M_MIN"] = [11.0, 10.0, 9.0]
    result = agg.aggregate_weather_window(frame, "2001-01-01", "2001-01-03")
    assert result["dry_days_lt1mm"] == 1
    assert result["rainy_days_ge1mm"] == 2
    assert result["heavy_rain_days_ge20mm"] == 1
    assert result["hot_days_tmax_ge35c"] == 2
    assert result["cold_days_tmin_lt10c"] == 1


def test_first_and_last_quarters_are_deterministic() -> None:
    frame = weather_frame(["2001-01-01", "2001-01-02", "2001-01-03", "2001-01-04", "2001-01-05"], [1.0, 2.0, 3.0, 4.0, 5.0])
    result = agg.aggregate_weather_window(frame, "2001-01-01", "2001-01-05")
    assert result["first_25pct_days"] == 2
    assert result["first_25pct_rain_sum_mm"] == 3.0
    assert result["last_25pct_days"] == 2
    assert result["last_25pct_rain_sum_mm"] == 9.0


def test_coverage_ratio_is_computed_from_present_days() -> None:
    result = agg.aggregate_weather_window(weather_frame(["2001-01-01", "2001-01-03"]), "2001-01-01", "2001-01-03")
    assert result["weather_days_expected"] == 3
    assert result["weather_days_present"] == 2
    assert result["weather_coverage_ratio"] == 2 / 3


def test_coverage_below_threshold_creates_invalid_window() -> None:
    result = agg.aggregate_weather_window(weather_frame(["2001-01-01", "2001-01-03"]), "2001-01-01", "2001-01-03")
    assert result["weather_window_valid"] is False
