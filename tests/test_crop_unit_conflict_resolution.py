from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import build_canonical_crop_weather_dataset as canonical  # noqa: E402
import diagnose_crop_unit_conflicts as diagnostic  # noqa: E402


def overlap_row(
    district: str = "PATIALA",
    legacy_area: float = 3.0,
    expanded_area: float = 3000.0,
    legacy_production: float = 264000.0,
    expanded_production: float = 264000.0,
) -> dict[str, object]:
    return {
        "canonical_state_name": "Punjab",
        "canonical_district_name": district,
        "Crop_Year": 2011,
        "Season_canonical": "Whole Year",
        "Crop_canonical": "Sugarcane",
        "legacy_source_row_id": 1,
        "expanded_source_row_id": 300000,
        "legacy_Area": legacy_area,
        "expanded_Area": expanded_area,
        "legacy_Production_corrected": legacy_production,
        "expanded_Production_corrected": expanded_production,
        "legacy_target_yield": legacy_production / legacy_area,
        "expanded_target_yield": expanded_production / expanded_area,
    }


def classify_one(row: dict[str, object]) -> pd.Series:
    frame = pd.DataFrame([row])
    classified = canonical.classify_conflict_pairs(frame, pd.Series([False], index=frame.index))
    return classified.iloc[0]


def prepared_row(
    row_id: int,
    source_name: str,
    district: str,
    area: float,
    production: float,
    year: int = 2011,
    crop: str = "Sugarcane",
) -> dict[str, object]:
    return {
        "source_row_id": row_id,
        "source_name": source_name,
        "canonical_state_name": "Punjab",
        "canonical_district_name": district,
        "Crop_Year": year,
        "Season_canonical": "Whole Year",
        "Crop_canonical": crop,
        "Area": area,
        "Production": production,
        "Area_corrected": area,
        "Production_corrected": production,
        "target_yield": production / area,
    }


def synthetic_prepared() -> pd.DataFrame:
    return pd.DataFrame(
        [
            prepared_row(1, "legacy_source", "PATIALA", 3.0, 264000.0),
            prepared_row(300000, "expanded_source_x100", "PATIALA", 3000.0, 264000.0),
            prepared_row(2, "legacy_source", "AMBIGUOUS", 10.0, 100.0),
            prepared_row(300001, "expanded_source_x100", "AMBIGUOUS", 12.0, 130.0),
            prepared_row(3, "legacy_source", "LEGACY ONLY", 5.0, 50.0),
        ]
    )


def test_same_production_and_area_times_1000_identifies_area_unit_conflict() -> None:
    row = classify_one(overlap_row())
    assert row["source_overlap_status"] == "conflict_unit_corrected"
    assert row["unit_correction_type"] == "area_unit_conflict"
    assert row["unit_correction_factor"] == 1000.0


def test_correct_source_row_is_selected_from_source_pair() -> None:
    row = classify_one(overlap_row())
    assert row["selected_source"] == "expanded_source_x100"


def test_rule_does_not_use_absolute_target_threshold() -> None:
    high_target_without_unit_evidence = overlap_row(
        legacy_area=3.0,
        expanded_area=4.0,
        legacy_production=264000.0,
        expanded_production=1000.0,
    )
    row = classify_one(high_target_without_unit_evidence)
    assert row["legacy_target_yield"] > 500
    assert row["source_overlap_status"] == "conflict_unresolved_legacy_retained"
    assert row["selected_source"] == "legacy_source"


def test_ambiguous_conflict_remains_unresolved() -> None:
    row = classify_one(overlap_row(legacy_area=10.0, expanded_area=12.0, legacy_production=100.0, expanded_production=130.0))
    assert row["source_overlap_status"] == "conflict_unresolved_legacy_retained"
    assert row["selected_source"] == "legacy_source"


def test_raw_values_are_preserved_in_conflict_report() -> None:
    canonical_frame, conflicts = canonical.reconcile_sources(synthetic_prepared())
    corrected = conflicts[conflicts["source_overlap_status"].eq("conflict_unit_corrected")].iloc[0]
    selected = canonical_frame[canonical_frame["canonical_district_name"].eq("PATIALA")].iloc[0]
    assert corrected["legacy_Area"] == 3.0
    assert corrected["expanded_Area"] == 3000.0
    assert selected["Area"] == 3000.0


def test_canonical_key_count_is_unchanged() -> None:
    canonical_frame, _ = canonical.reconcile_sources(synthetic_prepared())
    assert len(canonical_frame) == 3
    assert canonical_frame[canonical.CANONICAL_KEY_COLUMNS].drop_duplicates().shape[0] == 3


def test_punjab_sugarcane_2011_no_longer_creates_extreme_target() -> None:
    canonical_frame, _ = canonical.reconcile_sources(synthetic_prepared())
    patiala = canonical_frame[canonical_frame["canonical_district_name"].eq("PATIALA")].iloc[0]
    assert patiala["selected_source"] == "expanded_source_x100"
    assert patiala["target_yield"] == 88.0
    assert patiala["target_yield"] < 500


def test_diagnostic_does_not_read_test_dataset(monkeypatch) -> None:
    def fail_read_parquet(*args, **kwargs):
        raise AssertionError("diagnostic must not read parquet inputs")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)
    diagnosed = diagnostic.diagnose_conflicts(pd.DataFrame([overlap_row()]))
    assert len(diagnosed) == 1
