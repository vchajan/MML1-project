from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_canonical_crop_weather_dataset as canon  # noqa: E402


def base_row(
    source_row_id: int,
    state: str,
    district: str,
    year: int,
    season: str,
    crop: str,
    area: float,
    production: float,
    source_yield: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_row_id": source_row_id,
        "State_Name": state,
        "District_Name": district,
        "Crop_Year": year,
        "Season": season,
        "Crop": crop,
        "Area": area,
        "Production": production,
        "yield": source_yield,
        "start_date": f"{year}-06-01",
        "end_date": f"{year}-06-10",
        "district_id": f"DIST_{state}_{district}",
        "weather_point_id": f"WPT_{state}_{district}",
        "weather_window_id": f"WW_{source_row_id}",
        "latitude": 10.0,
        "longitude": 20.0,
        "point_assignment_confidence": "confirmed",
    }
    for column in canon.WEATHER_FEATURE_COLUMNS:
        row[column] = True if column == "weather_window_valid" else 1.0
    return row


def synthetic_input() -> pd.DataFrame:
    rows = [
        # Corroborated overlap: expanded values match legacy after factor 0.01.
        base_row(1, "Alpha", "One", 2000, " Kharif ", "Rice", 2.0, 10.0, 5.0),
        base_row(236379, "Alpha", "One", 2000, "Kharif", "Rice", 2.0, 1000.0, 500.0),
        # Conflict overlap: expanded corrected production differs.
        base_row(2, "Alpha", "Two", 2001, "Rabi", "Wheat", 3.0, 30.0, 10.0),
        base_row(236380, "Alpha", "Two", 2001, "Rabi", "Wheat", 3.0, 3300.0, 1100.0),
        # Legacy-only and expanded-only.
        base_row(3, "Alpha", "Three", 2002, "Summer", "Maize", 4.0, 16.0, 4.0),
        base_row(236381, "Alpha", "Four", 2003, "Whole Year", "Jute", 5.0, 2500.0, 500.0),
        # Manual district override and raw name preservation.
        base_row(4, "Punjab", "S", 2004, "Kharif", "Cotton(lint)", 5.0, 50.0, 10.0),
        # Complete dataset exclusions.
        base_row(5, "Beta", "Five", 2005, "Whole Year", "Coconut", 6.0, 60.0, 10.0),
        base_row(6, "Beta", "Six", 2006, "Whole Year", "Total foodgrain", 7.0, 70.0, 10.0),
        base_row(7, "Beta", "Seven", 2007, "Whole Year", "Pulses total", 8.0, 80.0, 10.0),
        base_row(8, "Beta", "Eight", 2008, "Whole Year", "Oilseeds total", 9.0, 90.0, 10.0),
    ]
    return pd.DataFrame(rows)


def overrides() -> list[canon.OverrideRule]:
    return [canon.OverrideRule("Punjab", "S", "Punjab", "S.A.S NAGAR")]


def canonical_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = canon.prepare_input(synthetic_input(), overrides())
    canonical, conflicts = canon.reconcile_sources(prepared)
    model_base = canon.build_model_base(canonical)
    return canonical, model_base, conflicts


def test_source_row_236378_is_legacy() -> None:
    assert canon.source_name_for_row_id(236378) == "legacy_source"


def test_source_row_236379_is_expanded() -> None:
    assert canon.source_name_for_row_id(236379) == "expanded_source_x100"


def test_expanded_production_is_scaled_by_001() -> None:
    prepared = canon.prepare_input(pd.DataFrame([base_row(236379, "A", "B", 2000, "Kharif", "Rice", 2.0, 1000.0, 500.0)]), [])
    assert prepared.loc[0, "Production_corrected"] == 10.0


def test_expanded_yield_is_scaled_by_001() -> None:
    prepared = canon.prepare_input(pd.DataFrame([base_row(236379, "A", "B", 2000, "Kharif", "Rice", 2.0, 1000.0, 500.0)]), [])
    assert prepared.loc[0, "yield_source_corrected"] == 5.0


def test_punjab_s_is_canonicalized_to_sas_nagar() -> None:
    prepared = canon.prepare_input(synthetic_input(), overrides())
    row = prepared[prepared["District_Name_raw"].eq("S")].iloc[0]
    assert row["canonical_state_name"] == "Punjab"
    assert row["canonical_district_name"] == "S.A.S NAGAR"
    assert row["district_override_applied"] == 1


def test_raw_name_is_preserved() -> None:
    prepared = canon.prepare_input(synthetic_input(), overrides())
    row = prepared[prepared["District_Name_raw"].eq("S")].iloc[0]
    assert row["District_Name_raw"] == "S"
    assert row["State_Name_raw"] == "Punjab"


def test_corroborated_overlap_selects_legacy() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    row = canonical[canonical["source_overlap_status"].eq("corroborated_after_scaling")].iloc[0]
    assert row["selected_source"] == "legacy_source"
    assert row["selected_source_row_id"] == 1


def test_unresolved_conflicting_overlap_selects_legacy() -> None:
    canonical, _model_base, conflicts = canonical_dataset()
    row = canonical[canonical["source_overlap_status"].eq("conflict_unresolved_legacy_retained")].iloc[0]
    assert row["selected_source"] == "legacy_source"
    assert row["selected_source_row_id"] == 2
    assert len(conflicts) == 1


def test_legacy_only_stays_unscaled() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    row = canonical[canonical["source_overlap_status"].eq("legacy_only") & canonical["Crop_canonical"].eq("Maize")].iloc[0]
    assert row["production_scale_factor"] == 1.0
    assert row["Production_corrected"] == 16.0


def test_expanded_only_uses_scaling() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    row = canonical[canonical["source_overlap_status"].eq("expanded_only_scaled")].iloc[0]
    assert row["selected_source"] == "expanded_source_x100"
    assert row["Production_corrected"] == 25.0


def test_target_is_always_corrected_production_over_area() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    calculated = canonical["Production_corrected"] / canonical["Area_corrected"]
    assert calculated.equals(canonical["target_yield"])


def test_canonical_key_is_unique() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    assert not canonical.duplicated(canon.CANONICAL_KEY_COLUMNS).any()


def test_coconut_remains_complete_but_not_model_base() -> None:
    canonical, model_base, _conflicts = canonical_dataset()
    assert canonical["Crop_canonical"].eq("Coconut").any()
    assert not model_base["Crop_canonical"].eq("Coconut").any()


def test_aggregate_crop_categories_are_not_model_base() -> None:
    canonical, model_base, _conflicts = canonical_dataset()
    assert canonical["Crop_canonical"].isin(canon.AGGREGATE_CROP_CATEGORIES).any()
    assert not model_base["Crop_canonical"].isin(canon.AGGREGATE_CROP_CATEGORIES).any()


def test_weather_features_are_preserved() -> None:
    canonical, _model_base, _conflicts = canonical_dataset()
    assert set(canon.WEATHER_FEATURE_COLUMNS).issubset(canonical.columns)
    assert int(canonical[canon.WEATHER_FEATURE_COLUMNS].isna().sum().sum()) == 0


def test_district_override_validation_accepts_already_reflected_input() -> None:
    prepared = canon.prepare_input(
        pd.DataFrame([base_row(1, "Punjab", "S.A.S NAGAR", 2004, "Kharif", "Rice", 5.0, 50.0, 10.0)]),
        overrides(),
    )
    canonical, _conflicts = canon.reconcile_sources(prepared)
    passed, detail = canon.district_override_validation(prepared, canonical, overrides())
    assert passed
    assert "reflected_rows=1" in detail


def test_expected_full_input_count_validates_to_canonical_count() -> None:
    stats = {name: value for name, value in canon.EXPECTED_COUNTS.items()}
    rows = canon.expected_count_validation_rows(stats)
    canonical_row = [row for row in rows if row["check"] == "canonical_rows"][0]
    input_row = [row for row in rows if row["check"] == "input_rows"][0]
    assert input_row["passed"]
    assert canonical_row["passed"]
    assert "observed=270300" in canonical_row["detail"]


def test_expected_model_base_count_validates() -> None:
    stats = {name: value for name, value in canon.EXPECTED_COUNTS.items()}
    rows = canon.expected_count_validation_rows(stats)
    model_row = [row for row in rows if row["check"] == "model_base_rows"][0]
    assert model_row["passed"]
    assert "observed=267150" in model_row["detail"]
