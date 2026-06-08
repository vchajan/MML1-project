from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CROP_PATH = REPO_ROOT / "data" / "raw" / "Indian_crop_production_yield_dataset.csv"
REQUIRED_DISTRICTS_PATH = REPO_ROOT / "data" / "reference" / "required_districts_1997_2014.csv"
CROSSWALK_TEMPLATE_PATH = REPO_ROOT / "data" / "reference" / "district_crosswalk_template_1997_2014.csv"
AUDIT_PATH = REPO_ROOT / "reports" / "district_name_audit.csv"
AUDIT_SUMMARY_PATH = REPO_ROOT / "reports" / "district_name_audit_summary.md"

ANOMALY_DETAILS_PATH = REPO_ROOT / "reports" / "district_name_anomaly_details.csv"
CANDIDATE_MATCHES_PATH = REPO_ROOT / "reports" / "district_name_candidate_matches.csv"
REVIEW_SUMMARY_PATH = REPO_ROOT / "reports" / "district_name_review_summary.md"
MANUAL_OVERRIDES_PATH = REPO_ROOT / "data" / "reference" / "district_name_manual_overrides.csv"

KEY_COLUMNS = ["Crop_Year", "Season", "Crop"]
VALUE_COLUMNS = ["Area", "Production", "yield"]
REQUIRED_RAW_COLUMNS = ["State_Name", "District_Name", *KEY_COLUMNS, *VALUE_COLUMNS]
REQUIRED_DISTRICT_COLUMNS = [
    "State_Name",
    "District_Name",
    "first_year",
    "last_year",
    "crop_rows",
    "unique_years",
    "unique_crops",
    "unique_seasons",
    "district_id",
]
REQUIRED_AUDIT_COLUMNS = [
    *REQUIRED_DISTRICT_COLUMNS,
    "district_compare_key",
    "review_priority",
    "flag_requires_manual_review",
    "issue_count",
]
FLAG_COLUMNS = [
    "flag_name_too_short",
    "flag_single_character_name",
    "flag_contains_digit",
    "flag_contains_parentheses",
    "flag_contains_unusual_punctuation",
    "flag_multiple_spaces_original",
    "flag_leading_or_trailing_space_original",
    "flag_duplicate_normalized_within_state",
    "flag_same_district_name_multiple_states",
    "flag_multiple_raw_names_same_normalized_key",
    "flag_possible_truncation",
]
OVERRIDE_COLUMNS = [
    "raw_state_name",
    "raw_district_name",
    "canonical_state_name",
    "canonical_district_name",
    "override_type",
    "evidence",
    "confidence",
    "review_status",
]
SPECIAL_PUNJAB_TERMS = ("S.A.S", "SAS", "SAHIBZADA", "NAGAR", "MOHALI")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["1", "true", "yes"])


def triggered_flags(row: pd.Series) -> list[str]:
    return [flag for flag in FLAG_COLUMNS if int(row.get(flag, 0)) == 1]


def primary_review_category(row: pd.Series) -> str:
    flags = set(triggered_flags(row))
    if flags & {"flag_duplicate_normalized_within_state", "flag_multiple_raw_names_same_normalized_key"}:
        return "actual_normalized_name_conflict"
    if flags & {"flag_name_too_short", "flag_single_character_name", "flag_possible_truncation"}:
        return "possible_damage_or_truncation"
    if "flag_contains_parentheses" in flags:
        return "parentheses"
    if "flag_same_district_name_multiple_states" in flags:
        return "possible_renaming_or_state_reorganization"
    if flags & {
        "flag_contains_digit",
        "flag_contains_unusual_punctuation",
        "flag_multiple_spaces_original",
        "flag_leading_or_trailing_space_original",
    }:
        return "punctuation_or_abbreviation"
    return "other_review"


def ratio_match_count(merged: pd.DataFrame, factor: float) -> int:
    if merged.empty:
        return 0

    def relation(left: pd.Series, right: pd.Series) -> pd.Series:
        valid = left.notna() & right.notna() & left.ne(0) & right.ne(0)
        ratio = pd.Series(False, index=left.index)
        ratio.loc[valid] = np.isclose(left.loc[valid] / right.loc[valid], factor) | np.isclose(
            right.loc[valid] / left.loc[valid], factor
        )
        return ratio

    production_relation = relation(merged["Production_problem"], merged["Production_candidate"])
    yield_relation = relation(merged["yield_problem"], merged["yield_candidate"])
    return int((production_relation | yield_relation).sum())


def compare_problem_to_candidate(
    problem_row: pd.Series,
    problem_raw: pd.DataFrame,
    candidate_district: str,
    candidate_raw: pd.DataFrame,
    district_id_lookup: dict[tuple[str, str], str],
) -> dict[str, object]:
    merged = problem_raw.merge(candidate_raw, on=KEY_COLUMNS, suffixes=("_problem", "_candidate"))
    same_area = merged["Area_problem"].eq(merged["Area_candidate"]) if not merged.empty else pd.Series(dtype=bool)
    same_production = (
        merged["Production_problem"].eq(merged["Production_candidate"]) if not merged.empty else pd.Series(dtype=bool)
    )
    same_yield = (
        np.isclose(merged["yield_problem"], merged["yield_candidate"], rtol=1e-9, atol=1e-9)
        if not merged.empty
        else np.array([], dtype=bool)
    )
    exact_match = same_area.to_numpy(dtype=bool) & same_production.to_numpy(dtype=bool) & np.asarray(same_yield)

    candidate_state = str(problem_row["State_Name"])
    candidate_id = district_id_lookup.get((candidate_state, candidate_district), "")
    same_key_count = int(len(merged))
    same_area_count = int(same_area.sum()) if not merged.empty else 0
    same_production_count = int(same_production.sum()) if not merged.empty else 0
    same_yield_count = int(np.asarray(same_yield).sum()) if not merged.empty else 0
    exact_count = int(exact_match.sum())
    ratio_10 = ratio_match_count(merged, 10)
    ratio_100 = ratio_match_count(merged, 100)
    ratio_1000 = ratio_match_count(merged, 1000)
    candidate_upper = candidate_district.upper()
    special_hint = any(term in candidate_upper for term in SPECIAL_PUNJAB_TERMS)
    problem_count = len(problem_raw)
    strong_scaled_match = (
        problem_count > 0
        and same_key_count >= int(problem_count * 0.95)
        and same_area_count >= int(problem_count * 0.7)
        and max(ratio_10, ratio_100, ratio_1000) >= int(problem_count * 0.9)
        and special_hint
    )

    if exact_count >= max(5, int(len(problem_raw) * 0.8)):
        candidate_status = "strong_data_match"
    elif strong_scaled_match:
        candidate_status = "strong_scaled_data_match"
    elif same_key_count and max(ratio_10, ratio_100, ratio_1000) >= max(5, int(same_key_count * 0.5)):
        candidate_status = "possible_scaling_relation"
    elif special_hint:
        candidate_status = "name_hint_only_no_data_confirmation"
    elif same_key_count:
        candidate_status = "reviewed_overlap_no_correction"
    else:
        candidate_status = "reviewed_no_overlap"

    evidence = (
        f"{same_key_count} shared Crop_Year+Season+Crop rows; "
        f"{same_area_count} same Area; {same_production_count} same Production; "
        f"{same_yield_count} same yield; {exact_count} exact data-row matches; "
        f"ratio hits x10/x100/x1000 = {ratio_10}/{ratio_100}/{ratio_1000}"
    )
    if special_hint:
        evidence += "; candidate name contains one of the requested Punjab review terms"

    return {
        "problem_district_id": problem_row["district_id"],
        "problem_state": problem_row["State_Name"],
        "problem_district": problem_row["District_Name"],
        "candidate_district_id": candidate_id,
        "candidate_state": candidate_state,
        "candidate_district": candidate_district,
        "same_year_season_crop_count": same_key_count,
        "same_area_count": same_area_count,
        "same_production_count": same_production_count,
        "same_yield_count": same_yield_count,
        "exact_data_row_match_count": exact_count,
        "production_ratio_10_count": ratio_10,
        "production_ratio_100_count": ratio_100,
        "production_ratio_1000_count": ratio_1000,
        "first_year": int(candidate_raw["Crop_Year"].min()) if not candidate_raw.empty else "",
        "last_year": int(candidate_raw["Crop_Year"].max()) if not candidate_raw.empty else "",
        "candidate_unique_crops": int(candidate_raw["Crop"].nunique()) if not candidate_raw.empty else 0,
        "candidate_unique_seasons": int(candidate_raw["Season"].nunique()) if not candidate_raw.empty else 0,
        "evidence_summary": evidence,
        "candidate_status": candidate_status,
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_CROP_PATH)
    required = pd.read_csv(REQUIRED_DISTRICTS_PATH)
    crosswalk = pd.read_csv(CROSSWALK_TEMPLATE_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    _ = AUDIT_SUMMARY_PATH.read_text(encoding="utf-8")

    require_columns(raw, REQUIRED_RAW_COLUMNS, "raw crop dataset")
    require_columns(required, REQUIRED_DISTRICT_COLUMNS, "required districts")
    require_columns(audit, REQUIRED_AUDIT_COLUMNS + FLAG_COLUMNS, "district name audit")
    if len(required) != 727:
        raise ValueError(f"Expected 727 required districts, found {len(required)}")
    if not required["district_id"].is_unique:
        raise ValueError("required_districts district_id is not unique")
    if not audit["district_id"].is_unique:
        raise ValueError("district_name_audit district_id is not unique")
    if "district_id" not in crosswalk.columns:
        raise ValueError("crosswalk template must contain district_id")

    return raw, required, crosswalk, audit


def build_candidate_matches(
    raw: pd.DataFrame, audit: pd.DataFrame, required: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    high_rows = audit[audit["review_priority"].eq("high")].copy()
    district_id_lookup = {
        (str(row.State_Name), str(row.District_Name)): str(row.district_id) for row in required.itertuples(index=False)
    }
    candidate_rows: list[dict[str, object]] = []
    candidate_counts: dict[str, int] = {}
    raw_1997_2014 = raw[raw["Crop_Year"].between(1997, 2014)].copy()

    for _, problem_row in high_rows.iterrows():
        state = str(problem_row["State_Name"])
        district = str(problem_row["District_Name"])
        problem_raw = raw_1997_2014[
            raw_1997_2014["State_Name"].eq(state) & raw_1997_2014["District_Name"].eq(district)
        ]
        same_state = raw_1997_2014[raw_1997_2014["State_Name"].eq(state)]
        candidates = sorted(d for d in same_state["District_Name"].dropna().unique() if d != district)
        candidate_counts[str(problem_row["district_id"])] = len(candidates)
        for candidate in candidates:
            candidate_raw = same_state[same_state["District_Name"].eq(candidate)]
            candidate_rows.append(
                compare_problem_to_candidate(problem_row, problem_raw, str(candidate), candidate_raw, district_id_lookup)
            )

    columns = [
        "problem_district_id",
        "problem_state",
        "problem_district",
        "candidate_district_id",
        "candidate_state",
        "candidate_district",
        "same_year_season_crop_count",
        "same_area_count",
        "same_production_count",
        "same_yield_count",
        "exact_data_row_match_count",
        "production_ratio_10_count",
        "production_ratio_100_count",
        "production_ratio_1000_count",
        "first_year",
        "last_year",
        "candidate_unique_crops",
        "candidate_unique_seasons",
        "evidence_summary",
        "candidate_status",
    ]
    return pd.DataFrame(candidate_rows, columns=columns), candidate_counts


def build_anomaly_details(audit: pd.DataFrame, candidate_counts: dict[str, int]) -> pd.DataFrame:
    flagged = audit[as_bool_series(audit["flag_requires_manual_review"])].copy()
    flagged["triggered_flags"] = flagged.apply(lambda row: ";".join(triggered_flags(row)), axis=1)
    flagged["review_category"] = flagged.apply(primary_review_category, axis=1)
    flagged["candidate_count"] = flagged["district_id"].map(candidate_counts).fillna(0).astype(int)

    conclusions: list[str] = []
    statuses: list[str] = []
    for _, row in flagged.iterrows():
        category = row["review_category"]
        if row["review_priority"] == "high":
            conclusions.append("possible damaged or truncated name; candidate review required data evidence")
            statuses.append("manual_external_verification_required")
        elif category == "possible_renaming_or_state_reorganization":
            conclusions.append("same normalized district name appears in multiple states; no correction proposed")
            statuses.append("reviewed_no_override")
        elif category == "parentheses":
            conclusions.append("parenthetical district name retained for later authoritative matching")
            statuses.append("reviewed_no_override")
        elif category == "punctuation_or_abbreviation":
            conclusions.append("punctuation or abbreviation retained; no data-backed correction proposed")
            statuses.append("reviewed_no_override")
        elif category == "actual_normalized_name_conflict":
            conclusions.append("normalized-name conflict requires manual review before crosswalk use")
            statuses.append("manual_review_required")
        else:
            conclusions.append("flag reviewed; no automatic correction proposed")
            statuses.append("reviewed_no_override")
    flagged["review_conclusion"] = conclusions
    flagged["review_status"] = statuses

    columns = [
        "district_id",
        "State_Name",
        "District_Name",
        "review_priority",
        "review_category",
        "triggered_flags",
        "first_year",
        "last_year",
        "crop_rows",
        "candidate_count",
        "review_conclusion",
        "review_status",
    ]
    return flagged[columns].sort_values(["review_priority", "State_Name", "District_Name"], ascending=[True, True, True])


def build_manual_overrides(candidate_matches: pd.DataFrame, audit: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    high_rows = audit[audit["review_priority"].eq("high")]
    for _, high in high_rows.iterrows():
        matches = candidate_matches[candidate_matches["problem_district_id"].eq(high["district_id"])].copy()
        strong = matches[matches["candidate_status"].isin(["strong_data_match", "strong_scaled_data_match"])].copy()
        strong["scale_evidence_count"] = 0
        if not strong.empty:
            strong["scale_evidence_count"] = strong[
                ["production_ratio_10_count", "production_ratio_100_count", "production_ratio_1000_count"]
            ].max(axis=1)
        strong = strong.sort_values(
            ["exact_data_row_match_count", "scale_evidence_count", "same_year_season_crop_count", "same_area_count"],
            ascending=False,
        )
        problem_raw = raw[
            raw["State_Name"].eq(high["State_Name"])
            & raw["District_Name"].eq(high["District_Name"])
            & raw["Crop_Year"].between(1997, 2014)
        ]

        if not strong.empty:
            best = strong.iloc[0]
            rows.append(
                {
                    "raw_state_name": high["State_Name"],
                    "raw_district_name": high["District_Name"],
                    "canonical_state_name": best["candidate_state"],
                    "canonical_district_name": best["candidate_district"],
                    "override_type": "truncated_name",
                    "evidence": best["evidence_summary"],
                    "confidence": "strong",
                    "review_status": "ready_for_crosswalk_override_review",
                }
            )
        elif high["State_Name"] == "Punjab" and high["District_Name"] == "S":
            best = matches.sort_values(
                ["exact_data_row_match_count", "same_year_season_crop_count", "same_area_count"],
                ascending=False,
            ).head(1)
            best_text = "no same-state candidate rows available"
            if not best.empty:
                best_row = best.iloc[0]
                best_text = (
                    f"best candidate `{best_row['candidate_district']}` had "
                    f"{best_row['exact_data_row_match_count']} exact data-row matches and "
                    f"{best_row['same_year_season_crop_count']} shared Crop_Year+Season+Crop rows"
                )
            rows.append(
                {
                    "raw_state_name": "Punjab",
                    "raw_district_name": "S",
                    "canonical_state_name": "",
                    "canonical_district_name": "",
                    "override_type": "data_corruption",
                    "evidence": (
                        f"Punjab/S has {len(problem_raw)} crop rows in 2006-2014; {best_text}; "
                        "no data-backed canonical district can be assigned without external verification"
                    ),
                    "confidence": "unresolved",
                    "review_status": "manual_external_verification_required",
                }
            )

    return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)


def build_summary(
    audit: pd.DataFrame,
    details: pd.DataFrame,
    candidate_matches: pd.DataFrame,
    overrides: pd.DataFrame,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> str:
    flagged = audit[as_bool_series(audit["flag_requires_manual_review"])].copy()
    flag_counts = {flag: int(flagged[flag].sum()) for flag in FLAG_COLUMNS}
    category_counts = details["review_category"].value_counts().sort_index()
    priority_counts = flagged["review_priority"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    normalized_conflict_rows = flagged[
        flagged["flag_duplicate_normalized_within_state"].eq(1)
        | flagged["flag_multiple_raw_names_same_normalized_key"].eq(1)
    ]

    punjab_matches = candidate_matches[candidate_matches["problem_state"].eq("Punjab") & candidate_matches["problem_district"].eq("S")]
    punjab_top = punjab_matches.sort_values(
        ["exact_data_row_match_count", "same_year_season_crop_count", "same_area_count"],
        ascending=False,
    ).head(8)
    punjab_special = punjab_matches[
        punjab_matches["candidate_district"].str.upper().apply(
            lambda value: any(term in value for term in SPECIAL_PUNJAB_TERMS)
        )
    ]
    punjab_override = overrides[overrides["raw_state_name"].eq("Punjab") & overrides["raw_district_name"].eq("S")]
    if punjab_override.empty:
        punjab_result = "No override row was created for Punjab / S."
    else:
        row = punjab_override.iloc[0]
        punjab_result = (
            f"`{row['confidence']}`; {row['review_status']}; evidence: {row['evidence']}"
        )

    lines: list[str] = [
        "# District Name Anomaly Review",
        "",
        "This review uses only local project inputs and exact data comparisons.",
        "No maps were downloaded, no geocoding was performed, no coordinates were assigned, "
        "and no fuzzy matching against external data was used.",
        "",
        "## Inputs",
        "",
        f"- Raw crop data SHA-256 before/after: `{hashes_before['raw']}` / `{hashes_after['raw']}`",
        f"- Required districts SHA-256 before/after: `{hashes_before['required']}` / `{hashes_after['required']}`",
        f"- Crosswalk template SHA-256 before/after: `{hashes_before['crosswalk']}` / `{hashes_after['crosswalk']}`",
        "",
        "## Flag Counts",
        "",
    ]
    lines.extend(f"- `{flag}`: {count}" for flag, count in flag_counts.items())
    lines.extend(
        [
            "",
            "## Review Categories",
            "",
        ]
    )
    lines.extend(f"- `{category}`: {int(count)}" for category, count in category_counts.items())
    lines.extend(
        [
            "",
            "## Priority Counts",
            "",
            f"- high: {int(priority_counts['high'])}",
            f"- medium: {int(priority_counts['medium'])}",
            f"- low: {int(priority_counts['low'])}",
            f"- normalized conflict rows: {len(normalized_conflict_rows)}",
            "",
            "## Punjab / S Review",
            "",
            f"- Result: {punjab_result}",
            "- Top same-state candidate comparisons:",
            "",
        ]
    )
    if punjab_top.empty:
        lines.append("- None.")
    else:
        for _, row in punjab_top.iterrows():
            lines.append(
                f"- `{row['candidate_district']}`: {row['same_year_season_crop_count']} shared keys, "
                f"{row['same_area_count']} same Area, {row['same_production_count']} same Production, "
                f"{row['same_yield_count']} same yield, {row['exact_data_row_match_count']} exact matches; "
                f"status `{row['candidate_status']}`"
            )
    lines.extend(["", "- Requested name-term checks:", ""])
    if punjab_special.empty:
        lines.append("- No Punjab candidates contained the requested terms.")
    else:
        for _, row in punjab_special.iterrows():
            lines.append(
                f"- `{row['candidate_district']}`: {row['same_year_season_crop_count']} shared keys, "
                f"{row['exact_data_row_match_count']} exact matches; status `{row['candidate_status']}`"
            )

    lines.extend(["", "## Manual Overrides", ""])
    if overrides.empty:
        lines.append("- No override rows created.")
    else:
        for _, row in overrides.iterrows():
            canonical = row["canonical_district_name"] if row["canonical_district_name"] else "<unresolved>"
            lines.append(
                f"- `{row['raw_state_name']}` / `{row['raw_district_name']}` -> `{canonical}`; "
                f"type `{row['override_type']}`; confidence `{row['confidence']}`; "
                f"status `{row['review_status']}`; evidence: {row['evidence']}"
            )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Raw crop dataset was not modified.",
            "- `required_districts_1997_2014.csv` was not modified.",
            "- All high-priority cases were analyzed.",
            "- Every override row has explicit evidence.",
            "- No district name was automatically corrected using fuzzy similarity.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs(
    raw: pd.DataFrame,
    audit: pd.DataFrame,
    details: pd.DataFrame,
    candidate_matches: pd.DataFrame,
    overrides: pd.DataFrame,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> None:
    flagged_count = int(as_bool_series(audit["flag_requires_manual_review"]).sum())
    if len(details) != flagged_count:
        raise ValueError(f"Expected {flagged_count} anomaly detail rows, found {len(details)}")

    high = audit[audit["review_priority"].eq("high")]
    analyzed_high_ids = set(candidate_matches["problem_district_id"].dropna().astype(str))
    missing_high = [district_id for district_id in high["district_id"].astype(str) if district_id not in analyzed_high_ids]
    if missing_high:
        raise ValueError(f"High-priority cases without candidate analysis: {missing_high}")

    if not overrides.empty:
        bad_override_type = sorted(set(overrides["override_type"]) - {"alias", "truncated_name", "renamed", "data_corruption"})
        bad_confidence = sorted(set(overrides["confidence"]) - {"confirmed", "strong", "unresolved"})
        if bad_override_type:
            raise ValueError(f"Invalid override_type values: {bad_override_type}")
        if bad_confidence:
            raise ValueError(f"Invalid confidence values: {bad_confidence}")
        if overrides["evidence"].isna().any() or overrides["evidence"].astype(str).str.strip().eq("").any():
            raise ValueError("Every override must have evidence")

    if hashes_before != hashes_after:
        raise ValueError("One or more protected input files changed during review")

    if raw.empty:
        raise ValueError("Raw crop dataset unexpectedly empty")


def main() -> int:
    hashes_before = {
        "raw": file_sha256(RAW_CROP_PATH),
        "required": file_sha256(REQUIRED_DISTRICTS_PATH),
        "crosswalk": file_sha256(CROSSWALK_TEMPLATE_PATH),
    }
    raw, required, _crosswalk, audit = load_inputs()

    candidate_matches, candidate_counts = build_candidate_matches(raw, audit, required)
    details = build_anomaly_details(audit, candidate_counts)
    overrides = build_manual_overrides(candidate_matches, audit, raw)

    ANOMALY_DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(ANOMALY_DETAILS_PATH, index=False, lineterminator="\n")
    candidate_matches.to_csv(CANDIDATE_MATCHES_PATH, index=False, lineterminator="\n")
    overrides.to_csv(MANUAL_OVERRIDES_PATH, index=False, lineterminator="\n")

    hashes_after = {
        "raw": file_sha256(RAW_CROP_PATH),
        "required": file_sha256(REQUIRED_DISTRICTS_PATH),
        "crosswalk": file_sha256(CROSSWALK_TEMPLATE_PATH),
    }
    validate_outputs(raw, audit, details, candidate_matches, overrides, hashes_before, hashes_after)
    REVIEW_SUMMARY_PATH.write_text(
        build_summary(audit, details, candidate_matches, overrides, hashes_before, hashes_after),
        encoding="utf-8",
        newline="\n",
    )

    flagged_count = len(details)
    confirmed_count = int(overrides["confidence"].eq("confirmed").sum()) if not overrides.empty else 0
    strong_count = int(overrides["confidence"].eq("strong").sum()) if not overrides.empty else 0
    unresolved_count = int(overrides["confidence"].eq("unresolved").sum()) if not overrides.empty else 0
    punjab = overrides[overrides["raw_state_name"].eq("Punjab") & overrides["raw_district_name"].eq("S")]
    punjab_result = "no override row"
    if not punjab.empty:
        row = punjab.iloc[0]
        punjab_result = f"{row['confidence']} / {row['review_status']}"

    print("District-name anomaly review completed")
    print(f"problematic_districts_analyzed={flagged_count}")
    print(f"candidate_pairs_analyzed={len(candidate_matches)}")
    print(f"punjab_s_result={punjab_result}")
    print(f"overrides_confirmed={confirmed_count}")
    print(f"overrides_strong={strong_count}")
    print(f"overrides_unresolved={unresolved_count}")
    print("raw_dataset_modified=false")
    print("required_districts_modified=false")
    print("fuzzy_external_matching_used=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"District-name anomaly review failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
