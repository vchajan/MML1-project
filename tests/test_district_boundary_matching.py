from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import match_districts_to_boundaries as matcher  # noqa: E402


def test_sas_nagar_normalization_removes_abbreviation_dots() -> None:
    assert matcher.compare_key("S.A.S NAGAR") == matcher.compare_key("SAS NAGAR")


def test_orissa_odisha_state_alias_is_explicit() -> None:
    assert matcher.is_state_alias("Odisha", "Orissa") == "explicit historical state alias ORISSA/ODISHA"


def test_exact_unique_match_confirms_only_single_candidate() -> None:
    row = pd.Series(
        {
            "district_id": "D1",
            "raw_state_name": "Example",
            "raw_district_name": "Alpha",
            "canonical_state_name": "Example",
            "canonical_district_name": "Alpha",
            "override_type": "",
        }
    )
    state_row = pd.Series(
        {
            "boundary_state_name": "Example",
            "match_status": "confirmed_by_name",
            "match_method": "unique_exact_normalized",
            "evidence": "test",
        }
    )
    inventory = pd.DataFrame(
        [
            {
                "census_version": "Census 2001",
                "source_feature_index": 1,
                "raw_state_name": "Example",
                "raw_district_name": "Alpha",
                "raw_state_code": "01",
                "raw_district_code": "001",
                "boundary_district_key": matcher.compare_key("Alpha"),
            }
        ]
    )
    stage_row, candidates = matcher.match_one(row, "Census 2001", state_row, inventory)
    assert stage_row["match_status"] == "matched_exact_unique"
    assert len(candidates) == 1


def test_multiple_exact_candidates_do_not_create_automatic_match() -> None:
    row = pd.Series(
        {
            "district_id": "D1",
            "raw_state_name": "Example",
            "raw_district_name": "Alpha",
            "canonical_state_name": "Example",
            "canonical_district_name": "Alpha",
            "override_type": "",
        }
    )
    state_row = pd.Series(
        {
            "boundary_state_name": "Example",
            "match_status": "confirmed_by_name",
            "match_method": "unique_exact_normalized",
            "evidence": "test",
        }
    )
    inventory = pd.DataFrame(
        [
            {
                "census_version": "Census 2001",
                "source_feature_index": idx,
                "raw_state_name": "Example",
                "raw_district_name": "Alpha",
                "raw_state_code": "01",
                "raw_district_code": str(idx),
                "boundary_district_key": matcher.compare_key("Alpha"),
            }
            for idx in [1, 2]
        ]
    )
    with pytest.raises(ValueError, match="more than one candidate"):
        matcher.match_one(row, "Census 2001", state_row, inventory)


def test_fuzzy_candidate_stays_unresolved() -> None:
    row = pd.Series(
        {
            "district_id": "D1",
            "raw_state_name": "Example",
            "raw_district_name": "Alpha",
            "canonical_state_name": "Example",
            "canonical_district_name": "Alpha",
            "override_type": "",
        }
    )
    state_row = pd.Series(
        {
            "boundary_state_name": "Example",
            "match_status": "confirmed_by_name",
            "match_method": "unique_exact_normalized",
            "evidence": "test",
        }
    )
    inventory = pd.DataFrame(
        [
            {
                "census_version": "Census 2001",
                "source_feature_index": 1,
                "raw_state_name": "Example",
                "raw_district_name": "Beta",
                "raw_state_code": "01",
                "raw_district_code": "001",
                "boundary_district_key": matcher.compare_key("Beta"),
            }
        ]
    )
    stage_row, candidates = matcher.match_one(row, "Census 2001", state_row, inventory)
    assert stage_row["match_status"] == "fuzzy_candidate_only"
    assert stage_row["match_confidence"] == "unresolved"
    assert candidates[0]["candidate_status"] == "fuzzy_candidate_only"


def test_punjab_s_uses_manual_override() -> None:
    stage2 = pd.read_csv(REPO_ROOT / "data" / "reference" / "district_boundary_crosswalk_stage2.csv")
    row = stage2[
        stage2["raw_state_name"].eq("Punjab")
        & stage2["raw_district_name"].eq("S")
        & stage2["census_version"].eq("Census 2011")
    ].iloc[0]
    assert row["canonical_district_name"] == "S.A.S NAGAR"
    assert row["match_status"] == "matched_manual_override"
    assert bool(row["manual_override_used"])


def test_raw_names_are_not_changed() -> None:
    stage1 = pd.read_csv(REPO_ROOT / "data" / "reference" / "district_crosswalk_stage1.csv")
    stage2 = pd.read_csv(REPO_ROOT / "data" / "reference" / "district_boundary_crosswalk_stage2.csv")
    stage2_raw = stage2[["district_id", "raw_state_name", "raw_district_name"]].drop_duplicates()
    merged = stage1[["district_id", "raw_state_name", "raw_district_name"]].merge(stage2_raw, on="district_id", suffixes=("_1", "_2"))
    assert len(merged) == 727
    assert (merged["raw_state_name_1"] == merged["raw_state_name_2"]).all()
    assert (merged["raw_district_name_1"] == merged["raw_district_name_2"]).all()


def test_status_report_has_727_rows() -> None:
    status = pd.read_csv(REPO_ROOT / "reports" / "district_boundary_matching_status.csv")
    assert len(status) == 727
