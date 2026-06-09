from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_canonical_crop_weather_dataset as canonical


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFLICTS_PATH = REPO_ROOT / "reports" / "crop_source_conflicts.csv"
PATTERNS_PATH = REPO_ROOT / "reports" / "crop_unit_conflict_patterns.csv"
DETAILS_PATH = REPO_ROOT / "reports" / "crop_unit_conflict_details.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "crop_unit_conflict_summary.md"

EXPECTED_CONFLICT_ROWS = 3_113


def diagnose_conflicts(conflicts: pd.DataFrame) -> pd.DataFrame:
    required = {
        *canonical.CANONICAL_KEY_COLUMNS,
        "legacy_source_row_id",
        "expanded_source_row_id",
        "legacy_Area",
        "expanded_Area",
        "legacy_Production_corrected",
        "expanded_Production_corrected",
        "legacy_target_yield",
        "expanded_target_yield",
    }
    canonical.require_columns(conflicts, required, "crop source conflict report")
    corroborated_mask = pd.Series(False, index=conflicts.index)
    diagnosed = canonical.classify_conflict_pairs(conflicts, corroborated_mask)
    diagnosed["corrected_productions_match"] = np.isclose(
        diagnosed["legacy_Production_corrected"],
        diagnosed["expanded_Production_corrected"],
        rtol=canonical.UNIT_RATIO_RTOL,
        atol=canonical.UNIT_PRODUCTION_MATCH_ATOL,
        equal_nan=False,
    )
    return diagnosed


def build_pattern_report(diagnosed: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "canonical_state_name",
        "Crop_Year",
        "Season_canonical",
        "Crop_canonical",
        "area_ratio_pattern",
        "production_ratio_pattern",
    ]
    grouped = (
        diagnosed.groupby(group_columns, dropna=False)
        .agg(
            row_count=("canonical_district_name", "size"),
            corrected_production_matches=("corrected_productions_match", "sum"),
            legacy_area_min=("legacy_Area", "min"),
            legacy_area_max=("legacy_Area", "max"),
            expanded_area_min=("expanded_Area", "min"),
            expanded_area_max=("expanded_Area", "max"),
            legacy_target_min=("legacy_target_yield", "min"),
            legacy_target_max=("legacy_target_yield", "max"),
            expanded_target_min=("expanded_target_yield", "min"),
            expanded_target_max=("expanded_target_yield", "max"),
            selected_sources=("selected_source", lambda values: ";".join(sorted(set(values.astype(str))))),
            source_overlap_statuses=(
                "source_overlap_status",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
        )
        .reset_index()
        .sort_values(["row_count", "canonical_state_name", "Crop_Year"], ascending=[False, True, True])
    )
    return grouped


def format_focus_districts(block: pd.DataFrame) -> list[str]:
    focus_names = ["PATIALA", "GURDASPUR", "TARN TARAN", "S.A.S NAGAR"]
    focus = block[block["canonical_district_name"].isin(focus_names)].copy()
    if focus.empty:
        return ["- No requested focus districts found in this block."]
    focus["corrected_target_yield"] = focus.apply(canonical.corrected_target_for_row, axis=1)
    lines = []
    for row in focus.sort_values("canonical_district_name").itertuples(index=False):
        lines.append(
            f"- {row.canonical_district_name}: legacy_area={row.legacy_Area:g}; "
            f"expanded_area={row.expanded_Area:g}; legacy_target={row.legacy_target_yield:g}; "
            f"expanded_target={row.expanded_target_yield:g}; corrected_target={row.corrected_target_yield:g}; "
            f"selected_source={row.selected_source}; status={row.source_overlap_status}"
        )
    return lines


def build_summary(diagnosed: pd.DataFrame, patterns: pd.DataFrame) -> str:
    punjab = canonical.target_block(diagnosed, "Punjab", 2011, "Whole Year", "Sugarcane")
    tamil_nadu = canonical.target_block(diagnosed, "Tamil Nadu", 1997, "Whole Year", "Sugarcane")
    lines = [
        "# Crop Unit Conflict Diagnostic Summary",
        "",
        f"- Conflict rows reviewed: {len(diagnosed)}",
        f"- Pattern groups: {len(patterns)}",
        f"- Area-unit corrections supported by deterministic source-pair evidence: {int(diagnosed['source_overlap_status'].eq('conflict_unit_corrected').sum())}",
        f"- Unresolved production-unit conflicts: {int(diagnosed['source_overlap_status'].eq('unresolved_production_unit_conflict').sum())}",
        f"- Unresolved conflicts with legacy retained: {int(diagnosed['source_overlap_status'].eq('conflict_unresolved_legacy_retained').sum())}",
        "",
        "No absolute target threshold, clipping, winsorization or row deletion is used.",
        "",
    ]
    lines.extend(canonical.block_summary_lines("Punjab / 2011 / Whole Year / Sugarcane", punjab))
    if not punjab.empty:
        lines.extend(["### Punjab Focus Districts", "", *format_focus_districts(punjab), ""])
    lines.extend(canonical.block_summary_lines("Tamil Nadu / 1997 / Whole Year / Sugarcane", tamil_nadu))
    return "\n".join(lines)


def write_diagnostics(conflicts_path: Path = CONFLICTS_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflicts = pd.read_csv(conflicts_path)
    if len(conflicts) != EXPECTED_CONFLICT_ROWS:
        raise ValueError(f"Expected {EXPECTED_CONFLICT_ROWS} conflict rows, found {len(conflicts)}")
    diagnosed = diagnose_conflicts(conflicts)
    patterns = build_pattern_report(diagnosed)
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(PATTERNS_PATH, index=False, lineterminator="\n")
    diagnosed.to_csv(DETAILS_PATH, index=False, lineterminator="\n")
    SUMMARY_PATH.write_text(build_summary(diagnosed, patterns), encoding="utf-8")
    return diagnosed, patterns


def main() -> int:
    diagnosed, patterns = write_diagnostics()
    print(f"conflict_rows={len(diagnosed)}")
    print(f"pattern_groups={len(patterns)}")
    print(f"unit_corrected_candidates={int(diagnosed['source_overlap_status'].eq('conflict_unit_corrected').sum())}")
    print(
        "unresolved_production_unit_conflicts="
        f"{int(diagnosed['source_overlap_status'].eq('unresolved_production_unit_conflict').sum())}"
    )
    print(
        "unresolved_legacy_retained_conflicts="
        f"{int(diagnosed['source_overlap_status'].eq('conflict_unresolved_legacy_retained').sum())}"
    )
    print(f"patterns_report={PATTERNS_PATH.relative_to(REPO_ROOT)}")
    print(f"details_report={DETAILS_PATH.relative_to(REPO_ROOT)}")
    print(f"summary_report={SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Crop unit conflict diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
