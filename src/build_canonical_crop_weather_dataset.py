from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_dataset_1997_2014.parquet"
RAW_CROP_PATH = REPO_ROOT / "data" / "raw" / "Indian_crop_production_yield_dataset.csv"
OVERRIDES_PATH = REPO_ROOT / "data" / "reference" / "district_name_manual_overrides.csv"
RULES_PATH = REPO_ROOT / "data" / "reference" / "crop_source_reconciliation_rules.json"

CANONICAL_OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_canonical_1997_2014.parquet"
MODEL_BASE_OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_model_base_1997_2014.parquet"

SUMMARY_PATH = REPO_ROOT / "reports" / "crop_source_reconciliation_summary.md"
CONFLICTS_PATH = REPO_ROOT / "reports" / "crop_source_conflicts.csv"
EXCLUSIONS_PATH = REPO_ROOT / "reports" / "crop_basic_model_exclusions.csv"
VALIDATION_PATH = REPO_ROOT / "reports" / "crop_canonical_dataset_validation.csv"
SAMPLE_PATH = REPO_ROOT / "reports" / "crop_canonical_dataset_sample.csv"

RAW_DATASET_SHA256 = "1A4651D07A271F882869109271610E6E9BD3B1870F3679AE0AC3AAACB728E5BC"
LEGACY_MAX_SOURCE_ROW_ID = 236_378
LEGACY_SCALE_FACTOR = 1.0
EXPANDED_SCALE_FACTOR = 0.01
CANONICAL_KEY_COLUMNS = [
    "canonical_state_name",
    "canonical_district_name",
    "Crop_Year",
    "Season_canonical",
    "Crop_canonical",
]
AGGREGATE_CROP_CATEGORIES = {"Total foodgrain", "Pulses total", "Oilseeds total"}
COCONUT_CROP = "Coconut"

WEATHER_FEATURE_COLUMNS = [
    "weather_days_expected",
    "weather_days_present",
    "weather_coverage_ratio",
    "weather_window_valid",
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
    "first_25pct_days",
    "first_25pct_rain_sum_mm",
    "first_25pct_temp_mean_c",
    "first_25pct_longest_dry_spell_days",
    "last_25pct_days",
    "last_25pct_rain_sum_mm",
    "last_25pct_temp_mean_c",
    "last_25pct_heavy_rain_days",
]

REQUIRED_INPUT_COLUMNS = {
    "source_row_id",
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production",
    "yield",
    "start_date",
    "end_date",
    "district_id",
    "weather_point_id",
    "weather_window_id",
    *WEATHER_FEATURE_COLUMNS,
}

EXPECTED_COUNTS = {
    "input_rows": 486_680,
    "legacy_rows": 235_817,
    "expanded_rows": 250_863,
    "canonical_rows": 270_300,
    "model_base_rows": 267_150,
    "legacy_only_keys": 19_437,
    "expanded_only_keys": 34_483,
    "overlapping_keys": 216_380,
    "corroborated_after_scaling": 213_267,
    "conflict_legacy_preferred": 3_113,
    "coconut_exclusions": 2_260,
    "total_foodgrain_exclusions": 188,
    "pulses_total_exclusions": 255,
    "oilseeds_total_exclusions": 447,
    "basic_exclusions": 3_150,
}


@dataclass(frozen=True)
class OverrideRule:
    raw_state_name: str
    raw_district_name: str
    canonical_state_name: str
    canonical_district_name: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def source_name_for_row_id(source_row_id: int) -> str:
    return "legacy_source" if int(source_row_id) <= LEGACY_MAX_SOURCE_ROW_ID else "expanded_source_x100"


def scale_factor_for_source(source_name: str) -> float:
    if source_name == "legacy_source":
        return LEGACY_SCALE_FACTOR
    if source_name == "expanded_source_x100":
        return EXPANDED_SCALE_FACTOR
    raise ValueError(f"Unknown source name: {source_name}")


def stable_id(prefix: str, parts: Iterable[object], length: int = 20) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:length].upper()}"


def canonical_key_string(frame: pd.DataFrame) -> pd.Series:
    return frame[CANONICAL_KEY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_override_rules(path: Path = OVERRIDES_PATH) -> list[OverrideRule]:
    overrides = pd.read_csv(path, keep_default_na=False)
    required = {"raw_state_name", "raw_district_name", "canonical_state_name", "canonical_district_name"}
    require_columns(overrides, required, "district overrides")
    duplicates = overrides.duplicated(["raw_state_name", "raw_district_name"], keep=False)
    if duplicates.any():
        rows = overrides.loc[duplicates, ["raw_state_name", "raw_district_name"]].to_dict(orient="records")
        raise ValueError(f"Duplicate district override rules: {rows}")
    return [
        OverrideRule(
            normalize_text(row.raw_state_name),
            normalize_text(row.raw_district_name),
            normalize_text(row.canonical_state_name),
            normalize_text(row.canonical_district_name),
        )
        for row in overrides.itertuples(index=False)
    ]


def apply_canonical_names(frame: pd.DataFrame, overrides: list[OverrideRule]) -> pd.DataFrame:
    result = frame.copy()
    result["State_Name_raw"] = result["State_Name"]
    result["District_Name_raw"] = result["District_Name"]
    result["Season_raw"] = result["Season"]
    result["Crop_raw"] = result["Crop"]

    result["canonical_state_name"] = result["State_Name"].map(normalize_text)
    result["canonical_district_name"] = result["District_Name"].map(normalize_text)
    result["Season_canonical"] = result["Season"].map(normalize_text)
    result["Crop_canonical"] = result["Crop"].map(normalize_text)
    result["district_override_applied"] = 0

    for rule in overrides:
        mask = result["canonical_state_name"].eq(rule.raw_state_name) & result["canonical_district_name"].eq(
            rule.raw_district_name
        )
        result.loc[mask, "canonical_state_name"] = rule.canonical_state_name
        result.loc[mask, "canonical_district_name"] = rule.canonical_district_name
        result.loc[mask, "district_override_applied"] = 1
    return result


def add_source_and_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source_name"] = result["source_row_id"].map(source_name_for_row_id)
    result["production_scale_factor"] = result["source_name"].map(scale_factor_for_source)
    result["Area_corrected"] = pd.to_numeric(result["Area"], errors="coerce")
    result["Production_corrected"] = pd.to_numeric(result["Production"], errors="coerce") * result["production_scale_factor"]
    result["yield_source_corrected"] = pd.to_numeric(result["yield"], errors="coerce") * result["production_scale_factor"]
    result["target_yield"] = result["Production_corrected"] / result["Area_corrected"]
    result["yield_formula_absolute_difference"] = (result["yield_source_corrected"] - result["target_yield"]).abs()
    result["_canonical_key"] = canonical_key_string(result)
    return result


def prepare_input(frame: pd.DataFrame, overrides: list[OverrideRule]) -> pd.DataFrame:
    require_columns(frame, REQUIRED_INPUT_COLUMNS, "crop-weather input")
    prepared = apply_canonical_names(frame, overrides)
    prepared = add_source_and_target_columns(prepared)
    return prepared


def validate_source_uniqueness(prepared: pd.DataFrame) -> None:
    duplicate_keys = prepared.duplicated(["source_name", *CANONICAL_KEY_COLUMNS], keep=False)
    if duplicate_keys.any():
        rows = prepared.loc[duplicate_keys, ["source_name", "source_row_id", *CANONICAL_KEY_COLUMNS]].head(10)
        raise ValueError(f"Each source must be unique by canonical key; duplicates include {rows.to_dict(orient='records')}")


def _is_close_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(np.isclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True), index=left.index)


def build_status_table(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = prepared[prepared["source_name"].eq("legacy_source")]
    expanded = prepared[prepared["source_name"].eq("expanded_source_x100")]

    legacy_keys = legacy[CANONICAL_KEY_COLUMNS + ["source_row_id", "Area_corrected", "Production_corrected", "target_yield"]].rename(
        columns={
            "source_row_id": "legacy_source_row_id",
            "Area_corrected": "legacy_Area",
            "Production_corrected": "legacy_Production_corrected",
            "target_yield": "legacy_target_yield",
        }
    )
    expanded_keys = expanded[
        CANONICAL_KEY_COLUMNS + ["source_row_id", "Area_corrected", "Production_corrected", "target_yield"]
    ].rename(
        columns={
            "source_row_id": "expanded_source_row_id",
            "Area_corrected": "expanded_Area",
            "Production_corrected": "expanded_Production_corrected",
            "target_yield": "expanded_target_yield",
        }
    )

    overlap = legacy_keys.merge(expanded_keys, on=CANONICAL_KEY_COLUMNS, how="inner", validate="one_to_one")
    area_match = _is_close_series(overlap["legacy_Area"], overlap["expanded_Area"])
    production_match = _is_close_series(
        overlap["legacy_Production_corrected"], overlap["expanded_Production_corrected"]
    )
    yield_match = _is_close_series(overlap["legacy_target_yield"], overlap["expanded_target_yield"])
    corroborated_mask = area_match & production_match & yield_match

    overlap_status = overlap.copy()
    overlap_status["source_rows_count"] = 2
    overlap_status["selected_source"] = "legacy_source"
    overlap_status["source_overlap_status"] = np.where(
        corroborated_mask,
        "corroborated_after_scaling",
        "conflict_legacy_preferred",
    )

    legacy_only = legacy_keys.merge(expanded_keys[CANONICAL_KEY_COLUMNS], on=CANONICAL_KEY_COLUMNS, how="left", indicator=True)
    legacy_only = legacy_only[legacy_only["_merge"].eq("left_only")].drop(columns=["_merge"])
    legacy_only["expanded_source_row_id"] = np.nan
    legacy_only["source_rows_count"] = 1
    legacy_only["selected_source"] = "legacy_source"
    legacy_only["source_overlap_status"] = "legacy_only"

    expanded_only = expanded_keys.merge(legacy_keys[CANONICAL_KEY_COLUMNS], on=CANONICAL_KEY_COLUMNS, how="left", indicator=True)
    expanded_only = expanded_only[expanded_only["_merge"].eq("left_only")].drop(columns=["_merge"])
    expanded_only["legacy_source_row_id"] = np.nan
    expanded_only["source_rows_count"] = 1
    expanded_only["selected_source"] = "expanded_source_x100"
    expanded_only["source_overlap_status"] = "expanded_only_scaled"

    status_columns = [
        *CANONICAL_KEY_COLUMNS,
        "source_rows_count",
        "legacy_source_row_id",
        "expanded_source_row_id",
        "selected_source",
        "source_overlap_status",
    ]
    status = pd.concat(
        [
            legacy_only[status_columns],
            expanded_only[status_columns],
            overlap_status[status_columns],
        ],
        ignore_index=True,
    )
    conflicts = build_conflict_report(overlap_status[overlap_status["source_overlap_status"].eq("conflict_legacy_preferred")])
    return status, conflicts


def build_conflict_report(conflicts: pd.DataFrame) -> pd.DataFrame:
    if conflicts.empty:
        return pd.DataFrame(
            columns=[
                *CANONICAL_KEY_COLUMNS,
                "legacy_source_row_id",
                "expanded_source_row_id",
                "legacy_Area",
                "expanded_Area",
                "legacy_Production_corrected",
                "expanded_Production_corrected",
                "legacy_target_yield",
                "expanded_target_yield",
                "area_difference",
                "production_difference",
                "yield_difference",
                "selected_source",
                "decision_reason",
            ]
        )
    report = conflicts.copy()
    report["area_difference"] = report["legacy_Area"] - report["expanded_Area"]
    report["production_difference"] = report["legacy_Production_corrected"] - report["expanded_Production_corrected"]
    report["yield_difference"] = report["legacy_target_yield"] - report["expanded_target_yield"]
    report["decision_reason"] = "legacy_source_preferred_for_conflicting_overlap"
    return report[
        [
            *CANONICAL_KEY_COLUMNS,
            "legacy_source_row_id",
            "expanded_source_row_id",
            "legacy_Area",
            "expanded_Area",
            "legacy_Production_corrected",
            "expanded_Production_corrected",
            "legacy_target_yield",
            "expanded_target_yield",
            "area_difference",
            "production_difference",
            "yield_difference",
            "selected_source",
            "decision_reason",
        ]
    ]


def reconcile_sources(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_source_uniqueness(prepared)
    status, conflicts = build_status_table(prepared)

    selected = prepared.merge(
        status[CANONICAL_KEY_COLUMNS + ["selected_source", "source_overlap_status"]],
        left_on=[*CANONICAL_KEY_COLUMNS, "source_name"],
        right_on=[*CANONICAL_KEY_COLUMNS, "selected_source"],
        how="inner",
        validate="one_to_one",
    )
    selected = selected.merge(
        status.drop(columns=["selected_source"]),
        on=[*CANONICAL_KEY_COLUMNS, "source_overlap_status"],
        how="left",
        validate="one_to_one",
    )
    selected["selected_source_row_id"] = selected["source_row_id"]
    selected["canonical_crop_row_id"] = selected.apply(
        lambda row: stable_id("CCR", [row[column] for column in CANONICAL_KEY_COLUMNS]),
        axis=1,
    )
    selected = add_model_eligibility(selected)

    if not selected["canonical_crop_row_id"].is_unique:
        raise ValueError("canonical_crop_row_id values are not unique")
    if selected.duplicated(CANONICAL_KEY_COLUMNS).any():
        raise ValueError("Canonical key is not unique after reconciliation")
    return selected, conflicts


def add_model_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_unit_group"] = "nominal_yield_per_area"
    result["basic_model_eligibility"] = 1
    result["basic_exclusion_reason"] = ""

    coconut_mask = result["Crop_canonical"].eq(COCONUT_CROP)
    result.loc[coconut_mask, "target_unit_group"] = "coconut_non_tonne_reporting_unit"
    result.loc[coconut_mask, "basic_model_eligibility"] = 0
    result.loc[coconut_mask, "basic_exclusion_reason"] = "incompatible_target_unit_coconut"

    aggregate_mask = result["Crop_canonical"].isin(AGGREGATE_CROP_CATEGORIES)
    result.loc[aggregate_mask, "target_unit_group"] = "aggregate_crop_category"
    result.loc[aggregate_mask, "basic_model_eligibility"] = 0
    result.loc[aggregate_mask, "basic_exclusion_reason"] = "aggregate_category_not_individual_crop"
    return result


def build_model_base(canonical: pd.DataFrame) -> pd.DataFrame:
    return canonical[canonical["basic_model_eligibility"].eq(1)].copy()


def compute_stats(input_frame: pd.DataFrame, canonical: pd.DataFrame, model_base: pd.DataFrame, conflicts: pd.DataFrame) -> dict[str, int | float]:
    status_counts = canonical["source_overlap_status"].value_counts()
    crop_counts = canonical["Crop_canonical"].value_counts()
    return {
        "input_rows": int(len(input_frame)),
        "legacy_rows": int(input_frame["source_name"].eq("legacy_source").sum()),
        "expanded_rows": int(input_frame["source_name"].eq("expanded_source_x100").sum()),
        "canonical_rows": int(len(canonical)),
        "model_base_rows": int(len(model_base)),
        "legacy_only_keys": int(status_counts.get("legacy_only", 0)),
        "expanded_only_keys": int(status_counts.get("expanded_only_scaled", 0)),
        "overlapping_keys": int(
            status_counts.get("corroborated_after_scaling", 0) + status_counts.get("conflict_legacy_preferred", 0)
        ),
        "corroborated_after_scaling": int(status_counts.get("corroborated_after_scaling", 0)),
        "conflict_legacy_preferred": int(status_counts.get("conflict_legacy_preferred", 0)),
        "coconut_exclusions": int(crop_counts.get(COCONUT_CROP, 0)),
        "total_foodgrain_exclusions": int(crop_counts.get("Total foodgrain", 0)),
        "pulses_total_exclusions": int(crop_counts.get("Pulses total", 0)),
        "oilseeds_total_exclusions": int(crop_counts.get("Oilseeds total", 0)),
        "basic_exclusions": int(canonical["basic_model_eligibility"].eq(0).sum()),
        "missing_weather_values": int(canonical[WEATHER_FEATURE_COLUMNS].isna().sum().sum()),
        "conflict_report_rows": int(len(conflicts)),
    }


def district_override_validation(prepared: pd.DataFrame, canonical: pd.DataFrame, overrides: list[OverrideRule]) -> tuple[bool, str]:
    selected_rows_with_override = int(canonical["district_override_applied"].sum())
    prepared_rows_with_override = int(prepared["district_override_applied"].sum())
    reflected_rows = 0
    raw_rule_rows = 0
    details = []
    for rule in overrides:
        raw_mask = prepared["State_Name_raw"].map(normalize_text).eq(rule.raw_state_name) & prepared[
            "District_Name_raw"
        ].map(normalize_text).eq(rule.raw_district_name)
        reflected_mask = prepared["canonical_state_name"].eq(rule.canonical_state_name) & prepared[
            "canonical_district_name"
        ].eq(rule.canonical_district_name)
        raw_count = int(raw_mask.sum())
        reflected_count = int(reflected_mask.sum())
        raw_rule_rows += raw_count
        reflected_rows += reflected_count
        details.append(
            f"{rule.raw_state_name}/{rule.raw_district_name}->{rule.canonical_state_name}/"
            f"{rule.canonical_district_name}: raw_rows={raw_count}, reflected_rows={reflected_count}"
        )

    passed = selected_rows_with_override > 0 or prepared_rows_with_override > 0 or reflected_rows > 0
    detail = (
        f"selected_rows_with_override={selected_rows_with_override}; "
        f"prepared_rows_with_override={prepared_rows_with_override}; "
        f"raw_rule_rows={raw_rule_rows}; reflected_rows={reflected_rows}; "
        + "; ".join(details)
    )
    return passed, detail


def expected_count_validation_rows(stats: dict[str, int | float], expected: dict[str, int] = EXPECTED_COUNTS) -> list[dict[str, object]]:
    rows = []
    for name, expected_value in expected.items():
        observed = int(stats.get(name, -1))
        rows.append(
            {
                "check": name,
                "passed": observed == expected_value,
                "detail": f"observed={observed}; expected={expected_value}",
            }
        )
    return rows


def build_validation_report(
    prepared: pd.DataFrame,
    canonical: pd.DataFrame,
    model_base: pd.DataFrame,
    stats: dict[str, int | float],
    raw_sha_before: str,
    raw_sha_after: str,
    overrides: list[OverrideRule],
) -> pd.DataFrame:
    override_passed, override_detail = district_override_validation(prepared, canonical, overrides)
    rows = [
        *expected_count_validation_rows(stats),
        {
            "check": "canonical_key_unique",
            "passed": not canonical.duplicated(CANONICAL_KEY_COLUMNS).any(),
            "detail": f"duplicate_keys={int(canonical.duplicated(CANONICAL_KEY_COLUMNS).sum())}",
        },
        {
            "check": "no_many_to_many_expansion",
            "passed": bool(canonical["source_rows_count"].isin([1, 2]).all()),
            "detail": f"max_source_rows_count={int(canonical['source_rows_count'].max())}",
        },
        {
            "check": "target_formula_valid",
            "passed": bool(
                np.isclose(
                    canonical["target_yield"],
                    canonical["Production_corrected"] / canonical["Area_corrected"],
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                ).all()
            ),
            "detail": "target_yield equals Production_corrected / Area_corrected",
        },
        {
            "check": "weather_features_preserved",
            "passed": set(WEATHER_FEATURE_COLUMNS).issubset(canonical.columns),
            "detail": f"weather_feature_columns={len([c for c in WEATHER_FEATURE_COLUMNS if c in canonical.columns])}",
        },
        {
            "check": "no_missing_weather_values",
            "passed": int(canonical[WEATHER_FEATURE_COLUMNS].isna().sum().sum()) == 0,
            "detail": f"missing_weather_values={int(canonical[WEATHER_FEATURE_COLUMNS].isna().sum().sum())}",
        },
        {
            "check": "district_override_applied",
            "passed": override_passed,
            "detail": override_detail,
        },
        {
            "check": "source_scaling_applied",
            "passed": bool(
                prepared.loc[prepared["source_name"].eq("expanded_source_x100"), "production_scale_factor"]
                .eq(EXPANDED_SCALE_FACTOR)
                .all()
            ),
            "detail": f"expanded_scale_factor={EXPANDED_SCALE_FACTOR}",
        },
        {
            "check": "raw_input_unchanged",
            "passed": raw_sha_before == raw_sha_after == RAW_DATASET_SHA256,
            "detail": f"before={raw_sha_before}; after={raw_sha_after}",
        },
    ]
    return pd.DataFrame(rows)


def validation_passed(validation: pd.DataFrame) -> bool:
    return bool(validation["passed"].all())


def build_sample(canonical: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for status in ["legacy_only", "expanded_only_scaled", "corroborated_after_scaling", "conflict_legacy_preferred"]:
        parts.append(canonical[canonical["source_overlap_status"].eq(status)].sort_values("canonical_crop_row_id").head(30))
    parts.append(canonical[canonical["Crop_canonical"].eq(COCONUT_CROP)].sort_values("canonical_crop_row_id").head(30))
    parts.append(canonical[canonical["Crop_canonical"].isin(AGGREGATE_CROP_CATEGORIES)].sort_values("canonical_crop_row_id").head(50))
    sample = pd.concat(parts, ignore_index=True).drop_duplicates("canonical_crop_row_id")
    return sample.head(200)


def write_rules_json(raw_sha: str) -> None:
    rules = {
        "raw_dataset_sha256": raw_sha,
        "source_split_source_row_id": {
            "legacy_source": f"source_row_id <= {LEGACY_MAX_SOURCE_ROW_ID}",
            "expanded_source_x100": f"source_row_id >= {LEGACY_MAX_SOURCE_ROW_ID + 1}",
        },
        "legacy_scale_factor": LEGACY_SCALE_FACTOR,
        "expanded_scale_factor": EXPANDED_SCALE_FACTOR,
        "canonical_key": CANONICAL_KEY_COLUMNS,
        "source_precedence": {
            "overlap": "legacy_source",
            "legacy_only": "legacy_source",
            "expanded_only": "expanded_source_x100",
        },
        "district_override_source": str(OVERRIDES_PATH.relative_to(REPO_ROOT)),
        "excluded_unit_categories": [COCONUT_CROP],
        "excluded_aggregate_categories": sorted(AGGREGATE_CROP_CATEGORIES),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")


def write_reports(
    canonical: pd.DataFrame,
    model_base: pd.DataFrame,
    conflicts: pd.DataFrame,
    validation: pd.DataFrame,
    stats: dict[str, int | float],
) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(CONFLICTS_PATH, index=False, lineterminator="\n")
    canonical[canonical["basic_model_eligibility"].eq(0)].to_csv(EXCLUSIONS_PATH, index=False, lineterminator="\n")
    validation.to_csv(VALIDATION_PATH, index=False, lineterminator="\n")
    build_sample(canonical).to_csv(SAMPLE_PATH, index=False, lineterminator="\n")

    lines = [
        "# Crop Source Reconciliation Summary",
        "",
        f"- Input rows: {stats['input_rows']}",
        f"- Legacy rows: {stats['legacy_rows']}",
        f"- Expanded rows: {stats['expanded_rows']}",
        f"- Canonical rows: {stats['canonical_rows']}",
        f"- Model-base rows: {stats['model_base_rows']}",
        "",
        "## Source Composition",
        "",
        f"- Legacy-only keys: {stats['legacy_only_keys']}",
        f"- Expanded-only keys: {stats['expanded_only_keys']}",
        f"- Overlapping keys: {stats['overlapping_keys']}",
        f"- Corroborated overlaps: {stats['corroborated_after_scaling']}",
        f"- Conflicting overlaps: {stats['conflict_legacy_preferred']}",
        "",
        "## Basic Model Exclusions",
        "",
        f"- Coconut: {stats['coconut_exclusions']}",
        f"- Total foodgrain: {stats['total_foodgrain_exclusions']}",
        f"- Pulses total: {stats['pulses_total_exclusions']}",
        f"- Oilseeds total: {stats['oilseeds_total_exclusions']}",
        f"- Total exclusions: {stats['basic_exclusions']}",
        "",
        "## Validation",
        "",
        f"- Missing weather values: {stats['missing_weather_values']}",
        f"- Validation checks passed: {validation_passed(validation)}",
        f"- Canonical parquet: {CANONICAL_OUTPUT_PATH.relative_to(REPO_ROOT)}",
        f"- Model-base parquet: {MODEL_BASE_OUTPUT_PATH.relative_to(REPO_ROOT)}",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    raw_sha_before = file_sha256(RAW_CROP_PATH)
    if raw_sha_before != RAW_DATASET_SHA256:
        raise ValueError(f"Raw crop dataset SHA-256 mismatch: {raw_sha_before}")

    overrides = load_override_rules(OVERRIDES_PATH)
    input_frame = pd.read_parquet(INPUT_PATH)
    prepared = prepare_input(input_frame, overrides)
    canonical, conflicts = reconcile_sources(prepared)
    model_base = build_model_base(canonical)
    stats = compute_stats(prepared, canonical, model_base, conflicts)
    raw_sha_after = file_sha256(RAW_CROP_PATH)
    validation = build_validation_report(prepared, canonical, model_base, stats, raw_sha_before, raw_sha_after, overrides)

    if not validation_passed(validation):
        VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(VALIDATION_PATH, index=False, lineterminator="\n")
        failures = validation.loc[~validation["passed"], ["check", "detail"]].to_dict(orient="records")
        raise ValueError(f"Canonical dataset validation failed: {failures}")

    CANONICAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(CANONICAL_OUTPUT_PATH, index=False)
    model_base.to_parquet(MODEL_BASE_OUTPUT_PATH, index=False)
    write_rules_json(raw_sha_before)
    write_reports(canonical, model_base, conflicts, validation, stats)

    print(f"input_rows={stats['input_rows']}")
    print(f"legacy_rows={stats['legacy_rows']}")
    print(f"expanded_rows={stats['expanded_rows']}")
    print(f"canonical_rows={stats['canonical_rows']}")
    print(f"model_base_rows={stats['model_base_rows']}")
    print(f"legacy_only_keys={stats['legacy_only_keys']}")
    print(f"expanded_only_keys={stats['expanded_only_keys']}")
    print(f"overlapping_keys={stats['overlapping_keys']}")
    print(f"corroborated_overlaps={stats['corroborated_after_scaling']}")
    print(f"conflicting_overlaps={stats['conflict_legacy_preferred']}")
    print(f"coconut_exclusions={stats['coconut_exclusions']}")
    print(f"basic_exclusions={stats['basic_exclusions']}")
    print(f"missing_weather_values={stats['missing_weather_values']}")
    print(f"canonical_output={CANONICAL_OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"model_base_output={MODEL_BASE_OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Canonical crop-weather build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
