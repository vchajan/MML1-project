from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggregate_crop_weather_windows import FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]
CROP_PATH = REPO_ROOT / "data" / "interim" / "crop_with_points_and_windows_1997_2014.parquet"
FEATURES_PATH = REPO_ROOT / "data" / "interim" / "weather_features_by_window_1997_2014.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_dataset_1997_2014.parquet"
SAMPLE_PATH = REPO_ROOT / "reports" / "crop_weather_dataset_sample.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "crop_weather_dataset_summary.md"
VALIDATION_PATH = REPO_ROOT / "reports" / "crop_weather_dataset_validation.csv"
EXPECTED_CROP_ROWS = 486_680


def validate_feature_table(features: pd.DataFrame) -> None:
    required = {"weather_window_id", *FEATURE_COLUMNS}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Weather features are missing columns: {', '.join(missing)}")
    duplicates = features["weather_window_id"].duplicated()
    if duplicates.any():
        duplicate_ids = features.loc[duplicates, "weather_window_id"].head(5).tolist()
        raise ValueError(f"weather_window_id must be unique in feature table; duplicates include {duplicate_ids}")


def build_joined_dataset(crop: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "weather_window_id" not in crop.columns:
        raise ValueError("Crop table is missing weather_window_id")
    if "source_row_id" not in crop.columns:
        raise ValueError("Crop table is missing source_row_id")
    if crop["weather_window_id"].isna().any():
        raise ValueError("Crop table contains missing weather_window_id")
    if not crop["source_row_id"].is_unique:
        raise ValueError("source_row_id must be unique before join")
    validate_feature_table(features)

    missing_window_ids = sorted(set(crop["weather_window_id"]) - set(features["weather_window_id"]))
    if missing_window_ids:
        preview = ", ".join(map(str, missing_window_ids[:5]))
        raise ValueError(f"Missing weather feature rows for weather_window_id values: {preview}")

    crop_ordered = crop.copy()
    crop_ordered["_input_order"] = range(len(crop_ordered))
    joined = crop_ordered.merge(
        features,
        on="weather_window_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_weather_feature"),
    ).sort_values("_input_order")
    joined = joined.drop(columns=["_input_order"])
    if len(joined) != len(crop):
        raise ValueError(f"Join changed row count: {len(crop)} -> {len(joined)}")
    if not joined["source_row_id"].equals(crop["source_row_id"].reset_index(drop=True)):
        raise ValueError("source_row_id order or values changed during join")

    metadata_pairs = [
        ("weather_point_id", "weather_point_id_weather_feature"),
        ("start_date", "start_date_weather_feature"),
        ("end_date", "end_date_weather_feature"),
    ]
    for crop_column, feature_column in metadata_pairs:
        if feature_column in joined.columns and crop_column in joined.columns:
            left = joined[crop_column].astype(str)
            right = joined[feature_column].astype(str)
            if left.ne(right).any():
                raise ValueError(f"Joined weather feature metadata does not match crop column {crop_column}")
            joined = joined.drop(columns=[feature_column])

    missing_features = joined[FEATURE_COLUMNS].isna().sum()
    if int(missing_features.sum()) > 0:
        missing_summary = ", ".join(f"{name}={int(count)}" for name, count in missing_features[missing_features > 0].items())
        raise ValueError(f"Joined dataset contains missing weather feature values: {missing_summary}")
    return joined


def build_validation(crop: pd.DataFrame, features: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("row_count_preserved", len(joined) == len(crop), f"{len(crop)} -> {len(joined)}"),
        ("no_many_to_many_expansion", len(joined) <= len(crop), f"joined_rows={len(joined)}"),
        (
            "all_weather_window_ids_matched",
            set(crop["weather_window_id"]).issubset(set(features["weather_window_id"])),
            f"crop_windows={crop['weather_window_id'].nunique()}, feature_windows={features['weather_window_id'].nunique()}",
        ),
        (
            "all_weather_features_present",
            int(joined[FEATURE_COLUMNS].isna().sum().sum()) == 0,
            f"missing_values={int(joined[FEATURE_COLUMNS].isna().sum().sum())}",
        ),
        (
            "source_row_id_preserved",
            joined["source_row_id"].equals(crop["source_row_id"].reset_index(drop=True)),
            f"unique_source_row_id={joined['source_row_id'].nunique()}",
        ),
        (
            "year_range_1997_2014",
            int(joined["Crop_Year"].min()) == 1997 and int(joined["Crop_Year"].max()) == 2014,
            f"{int(joined['Crop_Year'].min())}..{int(joined['Crop_Year'].max())}",
        ),
    ]
    return pd.DataFrame(
        [{"check": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]
    )


def write_reports(crop: pd.DataFrame, features: pd.DataFrame, joined: pd.DataFrame, validation: pd.DataFrame) -> None:
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.head(200).to_csv(SAMPLE_PATH, index=False, lineterminator="\n")
    validation.to_csv(VALIDATION_PATH, index=False, lineterminator="\n")

    missing_by_feature = joined[FEATURE_COLUMNS].isna().sum()
    missing_lines = [f"- Missing {column}: {int(count)}" for column, count in missing_by_feature.items()]
    confidence_column = "point_assignment_confidence"
    if confidence_column in joined.columns:
        confidence_counts = joined[confidence_column].value_counts(dropna=False).sort_index()
        confidence_lines = [f"- {label}: {int(count)}" for label, count in confidence_counts.items()]
    else:
        confidence_lines = ["- point_assignment_confidence column not found"]

    size_bytes = OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0
    lines = [
        "# Crop Weather Dataset Summary",
        "",
        f"- Input crop rows: {len(crop)}",
        f"- Output crop rows: {len(joined)}",
        f"- Columns: {len(joined.columns)}",
        f"- States: {joined['State_Name'].nunique()}",
        f"- Districts: {joined['district_id'].nunique()}",
        f"- Crops: {joined['Crop'].nunique()}",
        f"- Seasons: {joined['Season'].nunique()}",
        f"- Crop years: {int(joined['Crop_Year'].min())} to {int(joined['Crop_Year'].max())}",
        f"- Weather points: {joined['weather_point_id'].nunique()}",
        f"- Weather windows: {joined['weather_window_id'].nunique()}",
        f"- Output parquet: {OUTPUT_PATH.relative_to(REPO_ROOT)}",
        f"- Output parquet size bytes: {size_bytes}",
        "",
        "## Missing Weather Feature Values",
        "",
        *missing_lines,
        "",
        "## Geographic Assignment Confidence",
        "",
        *confidence_lines,
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    crop = pd.read_parquet(CROP_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    if len(crop) != EXPECTED_CROP_ROWS:
        raise ValueError(f"Expected {EXPECTED_CROP_ROWS} crop rows, found {len(crop)}")
    joined = build_joined_dataset(crop, features)
    if len(joined) != EXPECTED_CROP_ROWS:
        raise ValueError(f"Expected {EXPECTED_CROP_ROWS} output rows, found {len(joined)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUTPUT_PATH, index=False)
    validation = build_validation(crop, features, joined)
    write_reports(crop, features, joined, validation)
    if not validation["passed"].all():
        raise SystemExit("Crop weather dataset validation failed")

    print(f"input_crop_rows={len(crop)}")
    print(f"output_crop_rows={len(joined)}")
    print(f"columns={len(joined.columns)}")
    print(f"weather_windows={joined['weather_window_id'].nunique()}")
    print(f"missing_weather_values={int(joined[FEATURE_COLUMNS].isna().sum().sum())}")
    print(f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
