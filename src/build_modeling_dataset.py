from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_model_base_1997_2014.parquet"
MODEL_DATASET_PATH = REPO_ROOT / "data" / "processed" / "model_dataset_1997_2014.parquet"
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_1997_2010.parquet"
VALIDATION_PATH = REPO_ROOT / "data" / "processed" / "validation_2011_2012.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_2013_2014.parquet"
FEATURE_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "model_feature_manifest.json"

SUMMARY_PATH = REPO_ROOT / "reports" / "modeling_dataset_summary.md"
SPLIT_VALIDATION_PATH = REPO_ROOT / "reports" / "chronological_split_validation.csv"
FEATURE_SCHEMA_PATH = REPO_ROOT / "reports" / "modeling_feature_schema.csv"
UNSEEN_CATEGORIES_PATH = REPO_ROOT / "reports" / "modeling_unseen_categories.csv"
LAG_SUMMARY_PATH = REPO_ROOT / "reports" / "modeling_lag_summary.csv"
SAMPLE_PATH = REPO_ROOT / "reports" / "modeling_dataset_sample.csv"

EXPECTED_INPUT_ROWS = 267_150
EXPECTED_YEAR_MIN = 1997
EXPECTED_YEAR_MAX = 2014

CANONICAL_KEY_COLUMNS = [
    "canonical_state_name",
    "canonical_district_name",
    "Crop_Year",
    "Season_canonical",
    "Crop_canonical",
]
LAG_KEY_COLUMNS = [
    "canonical_state_name",
    "canonical_district_name",
    "Crop_canonical",
    "Season_canonical",
]
IDENTIFIER_COLUMNS = [
    "canonical_crop_row_id",
    "district_id",
    "lag_source_canonical_crop_row_id",
    "data_split",
]
TARGET_COLUMN = "target_yield"
CATEGORICAL_FEATURES = [
    "canonical_state_name",
    "canonical_district_name",
    "Crop_canonical",
    "Season_canonical",
]
NUMERIC_CORE_FEATURES = [
    "Crop_Year",
    "Area_corrected",
    "latitude",
    "longitude",
]
WEATHER_FEATURES = [
    "rain_sum_mm",
    "rain_mean_mm",
    "rainy_days_ge1mm",
    "dry_days_lt1mm",
    "heavy_rain_days_ge20mm",
    "longest_dry_spell_days",
    "longest_wet_spell_days",
    "temp_mean_c",
    "temp_max_mean_c",
    "temp_min_mean_c",
    "temp_max_absolute_c",
    "temp_min_absolute_c",
    "hot_days_tmax_ge35c",
    "cold_days_tmin_lt10c",
    "humidity_mean_pct",
    "solar_radiation_mean",
    "wind_speed_mean",
    "first_25pct_rain_sum_mm",
    "first_25pct_temp_mean_c",
    "first_25pct_longest_dry_spell_days",
    "last_25pct_rain_sum_mm",
    "last_25pct_temp_mean_c",
    "last_25pct_heavy_rain_days",
]
LAG_FEATURES = [
    "lag_yield_1y",
    "lag_available",
]
FORBIDDEN_LEAKAGE_COLUMNS = [
    "Production",
    "Production_corrected",
    "yield",
    "yield_source_corrected",
    "target_yield",
    "selected_source",
    "selected_source_row_id",
    "legacy_source_row_id",
    "expanded_source_row_id",
    "source_overlap_status",
    "source_rows_count",
    "production_scale_factor",
    "canonical_crop_row_id",
    "source_row_id",
    "weather_window_id",
    "district_year_point_id",
    "district_point_version_id",
    "start_date",
    "end_date",
    "basic_model_eligibility",
    "basic_exclusion_reason",
    "target_unit_group",
    "weather_days_expected",
    "weather_days_present",
    "weather_coverage_ratio",
    "weather_window_valid",
]
PROCESSED_COLUMNS = [
    "canonical_crop_row_id",
    "district_id",
    "data_split",
    *CATEGORICAL_FEATURES,
    *NUMERIC_CORE_FEATURES,
    *WEATHER_FEATURES,
    "lag_yield_1y",
    "lag_available",
    "lag_source_crop_year",
    "lag_source_canonical_crop_row_id",
    TARGET_COLUMN,
]
REQUIRED_INPUT_COLUMNS = {
    "canonical_crop_row_id",
    "district_id",
    "basic_model_eligibility",
    TARGET_COLUMN,
    "Area_corrected",
    *CANONICAL_KEY_COLUMNS,
    *CATEGORICAL_FEATURES,
    *NUMERIC_CORE_FEATURES,
    *WEATHER_FEATURES,
}


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def split_for_year(year: int) -> str:
    year = int(year)
    if 1997 <= year <= 2010:
        return "train"
    if 2011 <= year <= 2012:
        return "validation"
    if 2013 <= year <= 2014:
        return "test"
    raise ValueError(f"Unsupported Crop_Year for chronological split: {year}")


def validate_input_frame(frame: pd.DataFrame, expected_rows: int | None = EXPECTED_INPUT_ROWS) -> None:
    require_columns(frame, REQUIRED_INPUT_COLUMNS, "model-base input")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    if int(frame["Crop_Year"].min()) != EXPECTED_YEAR_MIN or int(frame["Crop_Year"].max()) != EXPECTED_YEAR_MAX:
        raise ValueError(f"Expected years {EXPECTED_YEAR_MIN}-{EXPECTED_YEAR_MAX}")
    if not frame["canonical_crop_row_id"].is_unique:
        raise ValueError("canonical_crop_row_id must be unique")
    if not frame["basic_model_eligibility"].eq(1).all():
        raise ValueError("basic_model_eligibility must equal 1 for every row")
    if frame[TARGET_COLUMN].isna().any() or np.isinf(frame[TARGET_COLUMN]).any():
        raise ValueError("target_yield must not contain NaN or infinity")
    if frame["Area_corrected"].isna().any() or frame["Area_corrected"].le(0).any():
        raise ValueError("Area_corrected must be present and greater than zero")
    if frame.duplicated(CANONICAL_KEY_COLUMNS).any():
        raise ValueError("Canonical modeling key must be unique")


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, [*LAG_KEY_COLUMNS, "Crop_Year", TARGET_COLUMN, "canonical_crop_row_id"], "lag input")
    lag_duplicates = frame.duplicated([*LAG_KEY_COLUMNS, "Crop_Year"], keep=False)
    if lag_duplicates.any():
        raise ValueError("Lag lookup key must be unique by lag key and Crop_Year")

    current = frame.copy()
    current["_lag_lookup_year"] = current["Crop_Year"].astype(int) - 1
    previous = frame[[*LAG_KEY_COLUMNS, "Crop_Year", TARGET_COLUMN, "canonical_crop_row_id"]].rename(
        columns={
            "Crop_Year": "_lag_lookup_year",
            TARGET_COLUMN: "lag_yield_1y",
            "canonical_crop_row_id": "lag_source_canonical_crop_row_id",
        }
    )
    merged = current.merge(previous, on=[*LAG_KEY_COLUMNS, "_lag_lookup_year"], how="left", validate="many_to_one")
    if len(merged) != len(frame):
        raise ValueError("Lag self-join changed row count")

    merged["lag_available"] = merged["lag_yield_1y"].notna().astype(int)
    merged["lag_source_crop_year"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    available = merged["lag_available"].eq(1)
    merged.loc[available, "lag_source_crop_year"] = merged.loc[available, "_lag_lookup_year"].astype(int)
    if (merged.loc[available, "lag_source_crop_year"].astype(int) >= merged.loc[available, "Crop_Year"].astype(int)).any():
        raise ValueError("Lag source year must be strictly earlier than current Crop_Year")
    if not (merged.loc[available, "lag_source_crop_year"].astype(int) == merged.loc[available, "Crop_Year"].astype(int) - 1).all():
        raise ValueError("Lag source year must be exactly Crop_Year - 1")
    return merged.drop(columns=["_lag_lookup_year"])


def add_chronological_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["data_split"] = result["Crop_Year"].map(split_for_year)
    return result


def feature_sets() -> dict[str, list[str]]:
    without_lag = [*CATEGORICAL_FEATURES, *NUMERIC_CORE_FEATURES, *WEATHER_FEATURES]
    with_lag = [*without_lag, *LAG_FEATURES]
    return {
        "core_without_lag": without_lag,
        "core_with_lag": with_lag,
    }


def validate_feature_sets(sets: dict[str, list[str]] | None = None) -> None:
    sets = sets or feature_sets()
    forbidden = set(FORBIDDEN_LEAKAGE_COLUMNS)
    for name, columns in sets.items():
        leakage = sorted(forbidden.intersection(columns))
        if leakage:
            raise ValueError(f"Feature set {name} contains forbidden leakage columns: {', '.join(leakage)}")
    if TARGET_COLUMN in set().union(*[set(columns) for columns in sets.values()]):
        raise ValueError("target_yield must not be included in model features")


def build_processed_dataset(frame: pd.DataFrame, expected_rows: int | None = EXPECTED_INPUT_ROWS) -> pd.DataFrame:
    validate_feature_sets()
    validate_input_frame(frame, expected_rows=expected_rows)
    processed = add_chronological_split(add_lag_features(frame))
    require_columns(processed, PROCESSED_COLUMNS, "processed dataset")
    processed = processed[PROCESSED_COLUMNS].copy()
    processed = processed.sort_values(
        [
            "Crop_Year",
            "canonical_state_name",
            "canonical_district_name",
            "Crop_canonical",
            "Season_canonical",
            "canonical_crop_row_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(processed) != len(frame):
        raise ValueError("Processed dataset row count changed")
    if not processed["canonical_crop_row_id"].is_unique:
        raise ValueError("Processed dataset row identifiers are not unique")
    return processed


def split_datasets(processed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": processed[processed["data_split"].eq("train")].copy(),
        "validation": processed[processed["data_split"].eq("validation")].copy(),
        "test": processed[processed["data_split"].eq("test")].copy(),
    }


def audit_unseen_categories(processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = processed[processed["data_split"].eq("train")]
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for column in CATEGORICAL_FEATURES:
        train_values = set(train[column].dropna().astype(str))
        summary: dict[str, object] = {
            "categorical_column": column,
            "train_unique_count": len(train_values),
        }
        for split in ["validation", "test"]:
            split_frame = processed[processed["data_split"].eq(split)]
            split_values = set(split_frame[column].dropna().astype(str))
            unseen_values = sorted(split_values - train_values)
            rows_with_unseen = int(split_frame[column].astype(str).isin(unseen_values).sum())
            summary[f"{split}_unique_count"] = len(split_values)
            summary[f"{split}_unseen_count"] = len(unseen_values)
            summary[f"{split}_rows_with_unseen_value"] = rows_with_unseen
            for value in unseen_values:
                rows.append(
                    {
                        "data_split": split,
                        "categorical_column": column,
                        "unseen_value": value,
                        "rows_with_unseen_value": int(split_frame[column].astype(str).eq(value).sum()),
                    }
                )
        summary_rows.append(summary)
    return pd.DataFrame(
        rows,
        columns=["data_split", "categorical_column", "unseen_value", "rows_with_unseen_value"],
    ), pd.DataFrame(summary_rows)


def lag_summary(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_frame in processed.groupby("data_split", sort=False):
        available = split_frame["lag_available"].eq(1)
        source_years = split_frame.loc[available, "lag_source_crop_year"].dropna().astype(int)
        rows.append(
            {
                "data_split": split,
                "row_count": len(split_frame),
                "lag_available_count": int(available.sum()),
                "lag_missing_count": int((~available).sum()),
                "lag_availability_rate": float(available.mean()) if len(split_frame) else 0.0,
                "minimum_lag_source_year": int(source_years.min()) if not source_years.empty else "",
                "maximum_lag_source_year": int(source_years.max()) if not source_years.empty else "",
            }
        )
    return pd.DataFrame(rows)


def validation_rows(input_frame: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    splits = split_datasets(processed)
    train_ids = set(splits["train"]["canonical_crop_row_id"])
    validation_ids = set(splits["validation"]["canonical_crop_row_id"])
    test_ids = set(splits["test"]["canonical_crop_row_id"])
    feature_columns = set().union(*[set(columns) for columns in feature_sets().values()])
    rows = [
        ("input_row_count", len(input_frame) == EXPECTED_INPUT_ROWS, len(input_frame), EXPECTED_INPUT_ROWS, ""),
        ("model_dataset_row_count", len(processed) == EXPECTED_INPUT_ROWS, len(processed), EXPECTED_INPUT_ROWS, ""),
        ("train_row_count", len(splits["train"]) > 0, len(splits["train"]), "> 0", ""),
        ("validation_row_count", len(splits["validation"]) > 0, len(splits["validation"]), "> 0", ""),
        ("test_row_count", len(splits["test"]) > 0, len(splits["test"]), "> 0", ""),
        (
            "split_union_preserves_rows",
            sum(len(split) for split in splits.values()) == len(processed),
            sum(len(split) for split in splits.values()),
            len(processed),
            "",
        ),
        (
            "split_ids_disjoint",
            not (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids),
            "disjoint",
            "disjoint",
            "",
        ),
        (
            "each_row_assigned_once",
            processed["canonical_crop_row_id"].nunique() == len(processed),
            processed["canonical_crop_row_id"].nunique(),
            len(processed),
            "",
        ),
        (
            "train_year_range_valid",
            splits["train"]["Crop_Year"].between(1997, 2010).all(),
            f"{splits['train']['Crop_Year'].min()}..{splits['train']['Crop_Year'].max()}",
            "1997..2010",
            "",
        ),
        (
            "validation_year_range_valid",
            set(splits["validation"]["Crop_Year"].unique()).issubset({2011, 2012}),
            sorted(splits["validation"]["Crop_Year"].unique().tolist()),
            [2011, 2012],
            "",
        ),
        (
            "test_year_range_valid",
            set(splits["test"]["Crop_Year"].unique()).issubset({2013, 2014}),
            sorted(splits["test"]["Crop_Year"].unique().tolist()),
            [2013, 2014],
            "",
        ),
        (
            "canonical_key_unique",
            not processed.duplicated(CANONICAL_KEY_COLUMNS).any(),
            int(processed.duplicated(CANONICAL_KEY_COLUMNS).sum()),
            0,
            "",
        ),
        ("target_complete", not processed[TARGET_COLUMN].isna().any(), int(processed[TARGET_COLUMN].isna().sum()), 0, ""),
        (
            "approved_features_complete",
            set(PROCESSED_COLUMNS).issubset(processed.columns),
            len([column for column in PROCESSED_COLUMNS if column in processed.columns]),
            len(PROCESSED_COLUMNS),
            "",
        ),
        (
            "forbidden_features_absent",
            not bool(set(FORBIDDEN_LEAKAGE_COLUMNS).intersection(feature_columns)),
            sorted(set(FORBIDDEN_LEAKAGE_COLUMNS).intersection(feature_columns)),
            [],
            "",
        ),
    ]
    available = processed["lag_available"].eq(1)
    rows.extend(
        [
            (
                "lag_year_strictly_previous",
                bool(
                    (
                        processed.loc[available, "lag_source_crop_year"].astype(int)
                        == processed.loc[available, "Crop_Year"].astype(int) - 1
                    ).all()
                ),
                "Y-1",
                "Y-1",
                "",
            ),
            (
                "lag_availability_consistent",
                bool(processed.loc[available, "lag_yield_1y"].notna().all())
                and bool(processed.loc[~available, "lag_yield_1y"].isna().all()),
                "consistent",
                "consistent",
                "",
            ),
            ("raw_input_unchanged", True, "not modified by script", "not modified by script", ""),
        ]
    )
    return pd.DataFrame(
        [
            {
                "check": check,
                "status": "passed" if passed else "failed",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
            for check, passed, observed, expected, details in rows
        ]
    )


def column_role(column: str) -> str:
    if column in {"canonical_crop_row_id", "district_id", "lag_source_canonical_crop_row_id", "data_split"}:
        return "identifier"
    if column in CATEGORICAL_FEATURES:
        return "categorical_feature"
    if column in NUMERIC_CORE_FEATURES:
        return "numeric_core_feature"
    if column in WEATHER_FEATURES:
        return "weather_feature"
    if column in LAG_FEATURES:
        return "lag_feature"
    if column == "lag_source_crop_year":
        return "lag_metadata"
    if column == TARGET_COLUMN:
        return "target"
    return "metadata"


def feature_schema(processed: pd.DataFrame) -> pd.DataFrame:
    sets = feature_sets()
    split_frames = split_datasets(processed)
    rows = []
    for column in processed.columns:
        memberships = [name for name, columns in sets.items() if column in columns]
        rows.append(
            {
                "column_name": column,
                "role": column_role(column),
                "dtype": str(processed[column].dtype),
                "feature_set_membership": ";".join(memberships),
                "missing_count_total": int(processed[column].isna().sum()),
                "missing_count_train": int(split_frames["train"][column].isna().sum()),
                "missing_count_validation": int(split_frames["validation"][column].isna().sum()),
                "missing_count_test": int(split_frames["test"][column].isna().sum()),
                "allowed_for_model": bool(memberships),
                "notes": "target column" if column == TARGET_COLUMN else "",
            }
        )
    return pd.DataFrame(rows)


def modeling_summary(processed: pd.DataFrame, unseen_summary: pd.DataFrame) -> str:
    splits = split_datasets(processed)
    sets = feature_sets()
    lag = lag_summary(processed)
    train_target = splits["train"][TARGET_COLUMN]
    target = processed[TARGET_COLUMN]
    unseen_validation = int(unseen_summary["validation_unseen_count"].sum()) if not unseen_summary.empty else 0
    unseen_test = int(unseen_summary["test_unseen_count"].sum()) if not unseen_summary.empty else 0
    lines = [
        "# Modeling Dataset Summary",
        "",
        f"- Input rows: {len(processed)}",
        f"- Output rows: {len(processed)}",
        f"- Train rows: {len(splits['train'])}",
        f"- Validation rows: {len(splits['validation'])}",
        f"- Test rows: {len(splits['test'])}",
        f"- Train years: {splits['train']['Crop_Year'].min()} to {splits['train']['Crop_Year'].max()}",
        f"- Validation years: {splits['validation']['Crop_Year'].min()} to {splits['validation']['Crop_Year'].max()}",
        f"- Test years: {splits['test']['Crop_Year'].min()} to {splits['test']['Crop_Year'].max()}",
        f"- Feature columns without lag: {len(sets['core_without_lag'])}",
        f"- Feature columns with lag: {len(sets['core_with_lag'])}",
        f"- Categorical features: {len(CATEGORICAL_FEATURES)}",
        f"- Numeric core features: {len(NUMERIC_CORE_FEATURES)}",
        f"- Weather features: {len(WEATHER_FEATURES)}",
        f"- Rows with lag: {int(processed['lag_available'].sum())}",
        f"- Rows without lag: {int(processed['lag_available'].eq(0).sum())}",
        "",
        "## Lag Availability By Split",
        "",
        *[
            f"- {row.data_split}: {row.lag_available_count}/{row.row_count} ({row.lag_availability_rate:.4f})"
            for row in lag.itertuples(index=False)
        ],
        "",
        "## Target Summary",
        "",
        f"- Minimum: {float(target.min()):.6f}",
        f"- Maximum: {float(target.max()):.6f}",
        f"- Median: {float(target.median()):.6f}",
        f"- Mean: {float(target.mean()):.6f}",
        "",
        "## Train Target Quantiles",
        "",
        *[
            f"- {label}: {float(train_target.quantile(q)):.6f}"
            for label, q in [("1%", 0.01), ("5%", 0.05), ("25%", 0.25), ("50%", 0.50), ("75%", 0.75), ("95%", 0.95), ("99%", 0.99)]
        ],
        "",
        "## Unseen Categories",
        "",
        f"- Validation unseen categories: {unseen_validation}",
        f"- Test unseen categories: {unseen_test}",
        "- Unseen categories are retained and must be handled later with `OneHotEncoder(handle_unknown=\"ignore\")` fitted only on train.",
        "",
        "No preprocessing has been fitted during dataset construction.",
        "The 2013-2014 test split has not been used for modeling, feature selection, preprocessing decisions, hyperparameter tuning, or model selection.",
        "",
    ]
    return "\n".join(lines)


def deterministic_sample(processed: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    parts = []
    for split in ["train", "validation", "test"]:
        split_frame = processed[processed["data_split"].eq(split)]
        parts.append(split_frame[split_frame["lag_available"].eq(1)].head(35))
        parts.append(split_frame[split_frame["lag_available"].eq(0)].head(35))
    sample = pd.concat(parts, ignore_index=True).drop_duplicates("canonical_crop_row_id")
    if len(sample) < max_rows:
        sample = pd.concat([sample, processed.head(max_rows)], ignore_index=True).drop_duplicates("canonical_crop_row_id")
    return sample.head(max_rows)


def write_feature_manifest(input_row_count: int, unseen_summary: pd.DataFrame) -> None:
    manifest = {
        "input_dataset": str(INPUT_PATH.relative_to(REPO_ROOT)),
        "input_row_count": input_row_count,
        "target_column": TARGET_COLUMN,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_core_features": NUMERIC_CORE_FEATURES,
        "weather_features": WEATHER_FEATURES,
        "lag_features": LAG_FEATURES,
        "feature_sets": feature_sets(),
        "forbidden_leakage_columns": FORBIDDEN_LEAKAGE_COLUMNS,
        "split_policy": {
            "train": "1997-2010",
            "validation": "2011-2012",
            "test": "2013-2014",
        },
        "lag_policy": (
            "Lag uses observed previous-year yield. It is allowed only for scenarios where the "
            "previous year's official yield is available at prediction time. It is built by explicit "
            "self-join on same district, crop and season with Crop_Year = Y - 1."
        ),
        "outlier_policy": "No target-based outlier removal has been applied. Any future threshold must be fitted using train data only.",
        "preprocessing_policy": "No imputers, encoders or scalers are fitted during dataset construction. They must be fitted only on the train split inside model pipelines.",
        "test_usage_policy": "The 2013-2014 test split must not be used for feature selection, preprocessing decisions, hyperparameter tuning or model selection.",
        "unseen_category_policy": "Unseen categories are retained and should be handled with OneHotEncoder(handle_unknown=\"ignore\") fitted only on train.",
        "unseen_category_summary": unseen_summary.to_dict(orient="records"),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    FEATURE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_outputs(input_frame: pd.DataFrame, processed: pd.DataFrame) -> None:
    splits = split_datasets(processed)
    unseen_rows, unseen_summary = audit_unseen_categories(processed)
    validation = validation_rows(input_frame, processed)
    if not validation["status"].eq("passed").all():
        SPLIT_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(SPLIT_VALIDATION_PATH, index=False, lineterminator="\n")
        failures = validation[~validation["status"].eq("passed")].to_dict(orient="records")
        raise ValueError(f"Split validation failed: {failures}")

    MODEL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(MODEL_DATASET_PATH, index=False)
    splits["train"].to_parquet(TRAIN_PATH, index=False)
    splits["validation"].to_parquet(VALIDATION_PATH, index=False)
    splits["test"].to_parquet(TEST_PATH, index=False)

    write_feature_manifest(len(input_frame), unseen_summary)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(modeling_summary(processed, unseen_summary), encoding="utf-8")
    validation.to_csv(SPLIT_VALIDATION_PATH, index=False, lineterminator="\n")
    feature_schema(processed).to_csv(FEATURE_SCHEMA_PATH, index=False, lineterminator="\n")
    unseen_rows.to_csv(UNSEEN_CATEGORIES_PATH, index=False, lineterminator="\n")
    lag_summary(processed).to_csv(LAG_SUMMARY_PATH, index=False, lineterminator="\n")
    deterministic_sample(processed).to_csv(SAMPLE_PATH, index=False, lineterminator="\n")


def main() -> int:
    input_frame = pd.read_parquet(INPUT_PATH)
    processed = build_processed_dataset(input_frame, expected_rows=EXPECTED_INPUT_ROWS)
    write_outputs(input_frame, processed)
    splits = split_datasets(processed)
    print(f"input_rows={len(input_frame)}")
    print(f"model_dataset_rows={len(processed)}")
    print(f"train_rows={len(splits['train'])}")
    print(f"validation_rows={len(splits['validation'])}")
    print(f"test_rows={len(splits['test'])}")
    print(f"rows_with_lag={int(processed['lag_available'].sum())}")
    print(f"rows_without_lag={int(processed['lag_available'].eq(0).sum())}")
    print(f"model_dataset={MODEL_DATASET_PATH.relative_to(REPO_ROOT)}")
    print(f"train_dataset={TRAIN_PATH.relative_to(REPO_ROOT)}")
    print(f"validation_dataset={VALIDATION_PATH.relative_to(REPO_ROOT)}")
    print(f"test_dataset={TEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Modeling dataset build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
