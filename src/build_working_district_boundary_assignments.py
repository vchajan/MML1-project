from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2_PATH = REPO_ROOT / "data" / "reference" / "district_boundary_crosswalk_stage2.csv"
CANDIDATES_PATH = REPO_ROOT / "reports" / "district_boundary_match_candidates.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "reference" / "district_boundary_assignments_working.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "working_boundary_assignment_summary.md"
STATUS_PATH = REPO_ROOT / "reports" / "working_boundary_assignment_status.csv"

CENSUS_VERSIONS = ["Census 2001", "Census 2011"]
DIRECT_MATCH_STATUS = {"matched_exact_unique", "matched_alias_unique", "matched_manual_override"}
CONFIRMED_METHOD = {
    "matched_exact_unique": "confirmed_exact_unique",
    "matched_alias_unique": "confirmed_alias_unique",
    "matched_manual_override": "confirmed_manual_override",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage2 = pd.read_csv(STAGE2_PATH, keep_default_na=False)
    candidates = pd.read_csv(CANDIDATES_PATH, keep_default_na=False)
    if len(stage2["district_id"].drop_duplicates()) != 727:
        raise ValueError("Stage2 must contain 727 district_id values")
    if len(stage2) != 1454:
        raise ValueError(f"Stage2 must contain 1454 rows, found {len(stage2)}")
    return stage2, candidates


def candidate_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candidate_rank_num"] = pd.to_numeric(out["candidate_rank"], errors="coerce").fillna(999999)
    out["string_similarity_num"] = pd.to_numeric(out["string_similarity"], errors="coerce").fillna(-1)
    out["edit_similarity_num"] = pd.to_numeric(out["edit_similarity"], errors="coerce").fillna(-1)
    out["boundary_feature_index_num"] = pd.to_numeric(out["boundary_feature_index"], errors="coerce").fillna(999999)
    return out.sort_values(
        ["candidate_rank_num", "string_similarity_num", "edit_similarity_num", "boundary_feature_index_num"],
        ascending=[True, False, False, True],
    )


def nonempty(value: object) -> bool:
    return str(value).strip() != ""


def direct_assignment(row: pd.Series) -> dict[str, object] | None:
    if row["match_status"] not in DIRECT_MATCH_STATUS or not nonempty(row["boundary_feature_index"]):
        return None
    return {
        "geometry_census_version": row["census_version"],
        "boundary_feature_index": row["boundary_feature_index"],
        "boundary_state_name": row["boundary_state_name"],
        "boundary_district_name": row["boundary_district_name"],
        "boundary_state_code": row["boundary_state_code"],
        "boundary_district_code": row["boundary_district_code"],
        "assignment_method": CONFIRMED_METHOD[row["match_status"]],
        "assignment_confidence": "confirmed",
        "candidate_rank": 1,
        "string_similarity": 1.0,
        "edit_similarity": 1.0,
        "cross_state_used": int(row["match_status"] == "cross_state_historical_candidate"),
        "alternate_layer_used": 0,
        "notes": row["notes"],
    }


def best_candidate(candidates: pd.DataFrame, district_id: str, census_version: str, status_filter: str | None = None) -> pd.Series | None:
    subset = candidates[
        candidates["district_id"].eq(district_id)
        & candidates["census_version"].eq(census_version)
        & candidates["boundary_feature_index"].astype(str).str.strip().ne("")
    ]
    if status_filter:
        subset = subset[subset["candidate_status"].eq(status_filter)]
    if subset.empty:
        return None
    return candidate_sort_key(subset).iloc[0]


def candidate_assignment(candidate: pd.Series, requested_version: str, method: str, confidence: str, notes: str) -> dict[str, object]:
    return {
        "geometry_census_version": candidate["census_version"],
        "boundary_feature_index": candidate["boundary_feature_index"],
        "boundary_state_name": candidate["boundary_state_name"],
        "boundary_district_name": candidate["boundary_district_name"],
        "boundary_state_code": candidate["boundary_state_code"],
        "boundary_district_code": candidate["boundary_district_code"],
        "assignment_method": method,
        "assignment_confidence": confidence,
        "candidate_rank": candidate["candidate_rank"],
        "string_similarity": candidate["string_similarity"],
        "edit_similarity": candidate["edit_similarity"],
        "cross_state_used": int(method == "historical_state_boundary_fallback"),
        "alternate_layer_used": int(candidate["census_version"] != requested_version),
        "notes": notes,
    }


def build_assignment_for_row(row: pd.Series, stage2: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, object]:
    requested = row["census_version"]
    district_id = row["district_id"]
    direct = direct_assignment(row)
    if direct:
        selected = direct
    else:
        other_version = "Census 2011" if requested == "Census 2001" else "Census 2001"
        other = stage2[stage2["district_id"].eq(district_id) & stage2["census_version"].eq(other_version)].iloc[0]
        other_direct = direct_assignment(other)
        if other_direct:
            selected = other_direct
            selected["assignment_method"] = "alternate_layer_confirmed_name"
            selected["assignment_confidence"] = "working_strong"
            selected["alternate_layer_used"] = 1
            selected["notes"] = f"Requested {requested} used confirmed name match from {other_version}"
        else:
            same_state_candidate = best_candidate(candidates, district_id, requested, "fuzzy_candidate_only")
            if same_state_candidate is not None:
                selected = candidate_assignment(
                    same_state_candidate,
                    requested,
                    "working_fuzzy_same_state",
                    "working_fallback",
                    "Best ranked same-state fuzzy candidate; not historically confirmed",
                )
            else:
                historical_candidate = best_candidate(candidates, district_id, requested, "cross_state_historical_candidate")
                if historical_candidate is not None:
                    selected = candidate_assignment(
                        historical_candidate,
                        requested,
                        "historical_state_boundary_fallback",
                        "historical_fallback",
                        "Historical cross-state boundary candidate used as working fallback",
                    )
                else:
                    other_candidate = best_candidate(candidates, district_id, other_version)
                    if other_candidate is None:
                        raise ValueError(f"No boundary candidate available for {district_id} {requested}")
                    selected = candidate_assignment(
                        other_candidate,
                        requested,
                        "alternate_layer_best_candidate",
                        "working_fallback",
                        f"Requested {requested} used best available candidate from {other_version}",
                    )

    return {
        "district_id": district_id,
        "raw_state_name": row["raw_state_name"],
        "raw_district_name": row["raw_district_name"],
        "canonical_state_name": row["canonical_state_name"],
        "canonical_district_name": row["canonical_district_name"],
        "requested_census_version": requested,
        "geometry_census_version": selected["geometry_census_version"],
        "boundary_feature_index": selected["boundary_feature_index"],
        "boundary_state_name": selected["boundary_state_name"],
        "boundary_district_name": selected["boundary_district_name"],
        "boundary_state_code": selected["boundary_state_code"],
        "boundary_district_code": selected["boundary_district_code"],
        "assignment_method": selected["assignment_method"],
        "assignment_confidence": selected["assignment_confidence"],
        "candidate_rank": selected["candidate_rank"],
        "string_similarity": selected["string_similarity"],
        "edit_similarity": selected["edit_similarity"],
        "manual_override_used": int(str(row.get("manual_override_used", "")).lower() == "true"),
        "cross_state_used": selected["cross_state_used"],
        "alternate_layer_used": selected["alternate_layer_used"],
        "shared_parent_boundary": 0,
        "requires_future_manual_review": int(selected["assignment_confidence"] != "confirmed"),
        "notes": selected["notes"],
    }


def mark_shared_parent(assignments: pd.DataFrame) -> pd.DataFrame:
    out = assignments.copy()
    group_cols = ["requested_census_version", "geometry_census_version", "boundary_feature_index"]
    sizes = out.groupby(group_cols)["district_id"].transform("nunique")
    out["shared_parent_boundary"] = (sizes > 1).astype(int)
    return out


def build_assignments(stage2: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [build_assignment_for_row(row, stage2, candidates) for _, row in stage2.iterrows()]
    assignments = pd.DataFrame(rows)
    assignments = mark_shared_parent(assignments)
    if len(assignments) != 1454:
        raise ValueError(f"Expected 1454 assignment rows, found {len(assignments)}")
    if assignments["boundary_feature_index"].astype(str).str.strip().eq("").any():
        raise ValueError("Some assignments are missing boundary_feature_index")
    if assignments["geometry_census_version"].astype(str).str.strip().eq("").any():
        raise ValueError("Some assignments are missing geometry_census_version")
    counts = assignments.groupby("district_id")["requested_census_version"].nunique()
    if len(counts) != 727 or counts.min() != 2 or counts.max() != 2:
        raise ValueError("Each district_id must have Census 2001 and Census 2011 assignments")
    return assignments


def write_reports(assignments: pd.DataFrame) -> None:
    status = assignments.pivot_table(
        index=["district_id", "raw_state_name", "raw_district_name", "canonical_state_name", "canonical_district_name"],
        columns="requested_census_version",
        values=["assignment_method", "assignment_confidence", "geometry_census_version", "shared_parent_boundary"],
        aggfunc="first",
    )
    status.columns = [f"{metric}_{version.replace('Census ', '')}" for metric, version in status.columns]
    status = status.reset_index()
    if len(status) != 727:
        raise ValueError("Working boundary assignment status must have 727 rows")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(STATUS_PATH, index=False, lineterminator="\n")

    counts = assignments["assignment_confidence"].value_counts()
    method_counts = assignments["assignment_method"].value_counts()
    lines = [
        "# Working Boundary Assignment Summary",
        "",
        f"- District IDs: {assignments['district_id'].nunique()}",
        f"- Assignment rows: {len(assignments)}",
        f"- Confirmed assignments: {int(counts.get('confirmed', 0))}",
        f"- Working strong assignments: {int(counts.get('working_strong', 0))}",
        f"- Working fallback assignments: {int(counts.get('working_fallback', 0))}",
        f"- Historical fallback assignments: {int(counts.get('historical_fallback', 0))}",
        f"- Shared parent boundary rows: {int(assignments['shared_parent_boundary'].sum())}",
        f"- Future manual review rows: {int(assignments['requires_future_manual_review'].sum())}",
        "",
        "## Methods",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in method_counts.sort_index().items())
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    stage2, candidates = load_inputs()
    assignments = build_assignments(stage2, candidates)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")
    write_reports(assignments)
    print(f"assignment_rows={len(assignments)}")
    print(f"district_ids={assignments['district_id'].nunique()}")
    print(f"confirmed={int(assignments['assignment_confidence'].eq('confirmed').sum())}")
    print(f"working_fallback={int(assignments['assignment_confidence'].eq('working_fallback').sum())}")
    print(f"historical_fallback={int(assignments['assignment_confidence'].eq('historical_fallback').sum())}")
    print(f"shared_parent_boundary={int(assignments['shared_parent_boundary'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
