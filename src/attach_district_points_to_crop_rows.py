from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CROP_PATH = REPO_ROOT / "data" / "interim" / "crop_with_calendar_dates_1997_2014.csv"
REQUIRED_PATH = REPO_ROOT / "data" / "reference" / "required_districts_1997_2014.csv"
POINTS_BY_YEAR_PATH = REPO_ROOT / "data" / "reference" / "district_points_by_crop_year.csv"
OUTPUT_CROP_POINTS = REPO_ROOT / "data" / "interim" / "crop_with_points_and_windows_1997_2014.parquet"
OUTPUT_WINDOWS = REPO_ROOT / "data" / "interim" / "weather_windows_1997_2014.parquet"
SUMMARY_PATH = REPO_ROOT / "reports" / "crop_point_assignment_summary.md"
STATUS_PATH = REPO_ROOT / "reports" / "crop_point_assignment_status.csv"


def stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:length].upper()}"


def expected_days(start: pd.Series, end: pd.Series) -> pd.Series:
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.days + 1


def main() -> int:
    crop = pd.read_csv(CROP_PATH)
    required = pd.read_csv(REQUIRED_PATH)
    points = pd.read_csv(POINTS_BY_YEAR_PATH)
    input_rows = len(crop)
    if input_rows != 486680:
        raise ValueError(f"Expected 486680 crop rows, found {input_rows}")
    district_lookup = required[["State_Name", "District_Name", "district_id"]].drop_duplicates()
    if len(district_lookup) != 727:
        raise ValueError("District lookup must contain 727 rows")
    crop = crop.merge(district_lookup, on=["State_Name", "District_Name"], how="left", validate="many_to_one")
    if crop["district_id"].isna().any():
        raise ValueError("Some crop rows did not receive district_id")
    crop = crop.merge(points, on=["district_id", "Crop_Year"], how="left", validate="many_to_one")
    if len(crop) != input_rows:
        raise ValueError("Crop-point merge changed row count")
    if crop["district_year_point_id"].isna().any() or crop["latitude"].isna().any() or crop["longitude"].isna().any():
        raise ValueError("Some crop rows did not receive a point")
    if crop.duplicated().sum():
        raise ValueError("Unexpected duplicated full crop rows after point merge")
    crop["weather_window_id"] = [
        stable_id("WW", point_id, start, end)
        for point_id, start, end in zip(crop["weather_point_id"], crop["start_date"], crop["end_date"])
    ]
    if crop["weather_window_id"].isna().any():
        raise ValueError("Some crop rows did not receive weather_window_id")
    crop = crop.rename(
        columns={
            "assignment_method": "point_assignment_method",
            "assignment_confidence": "point_assignment_confidence",
        }
    )
    windows = (
        crop.groupby(["weather_window_id", "weather_point_id", "latitude", "longitude", "start_date", "end_date"])
        .size()
        .reset_index(name="crop_rows_count")
    )
    windows["expected_days"] = expected_days(windows["start_date"], windows["end_date"])
    windows = windows[
        [
            "weather_window_id",
            "weather_point_id",
            "latitude",
            "longitude",
            "start_date",
            "end_date",
            "expected_days",
            "crop_rows_count",
        ]
    ]
    OUTPUT_CROP_POINTS.parent.mkdir(parents=True, exist_ok=True)
    crop.to_parquet(OUTPUT_CROP_POINTS, index=False)
    windows.to_parquet(OUTPUT_WINDOWS, index=False)
    status = crop.groupby(["point_assignment_confidence", "point_assignment_method"]).size().reset_index(name="crop_rows")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(STATUS_PATH, index=False, lineterminator="\n")
    lines = [
        "# Crop Point Assignment Summary",
        "",
        f"- Input crop rows: {input_rows}",
        f"- Output crop rows: {len(crop)}",
        f"- Rows without district_id: {int(crop['district_id'].isna().sum())}",
        f"- Rows without point: {int(crop['district_year_point_id'].isna().sum())}",
        f"- Unique weather windows: {len(windows)}",
        f"- Unique weather points: {crop['weather_point_id'].nunique()}",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"input_crop_rows={input_rows}")
    print(f"output_crop_rows={len(crop)}")
    print(f"rows_without_point={int(crop['district_year_point_id'].isna().sum())}")
    print(f"unique_weather_windows={len(windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
