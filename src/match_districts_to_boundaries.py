from __future__ import annotations

import difflib
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_PATH = REPO_ROOT / "data" / "reference" / "district_crosswalk_stage1.csv"
MANUAL_OVERRIDES_PATH = REPO_ROOT / "data" / "reference" / "district_name_manual_overrides.csv"
BOUNDARY_INVENTORY_PATH = REPO_ROOT / "reports" / "boundary_district_name_inventory.csv"
LAYER_AUDIT_PATH = REPO_ROOT / "reports" / "district_boundary_layer_audit.csv"
SOURCE_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "boundary_sources" / "datameet_district_boundaries.json"

STATE_CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "state_boundary_crosswalk.csv"
STAGE2_PATH = REPO_ROOT / "data" / "reference" / "district_boundary_crosswalk_stage2.csv"
CANDIDATES_PATH = REPO_ROOT / "reports" / "district_boundary_match_candidates.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "district_boundary_matching_summary.md"
STATUS_PATH = REPO_ROOT / "reports" / "district_boundary_matching_status.csv"

EXPECTED_STAGE1_ROWS = 727
EXPECTED_FEATURE_COUNTS = {"Census 2001": 594, "Census 2011": 641}
CENSUS_VERSIONS = ["Census 2001", "Census 2011"]
MATCHED_STATUSES = {"matched_exact_unique", "matched_alias_unique", "matched_manual_override"}
ALLOWED_MATCH_STATUSES = {
    "matched_exact_unique",
    "matched_alias_unique",
    "matched_manual_override",
    "multiple_candidates",
    "fuzzy_candidate_only",
    "cross_state_historical_candidate",
    "unmatched",
}
ALLOWED_CONFIDENCE = {"confirmed_by_name", "strong_candidate", "unresolved"}


STATE_ALIASES = {
    ("ODISHA", "ORISSA"): "explicit historical state alias ORISSA/ODISHA",
    ("ORISSA", "ODISHA"): "explicit historical state alias ORISSA/ODISHA",
    ("UTTARAKHAND", "UTTARANCHAL"): "explicit historical state alias UTTARANCHAL/UTTARAKHAND",
    ("UTTARANCHAL", "UTTARAKHAND"): "explicit historical state alias UTTARANCHAL/UTTARAKHAND",
    ("PUDUCHERRY", "PONDICHERRY"): "explicit historical state alias PONDICHERRY/PUDUCHERRY",
    ("PONDICHERRY", "PUDUCHERRY"): "explicit historical state alias PONDICHERRY/PUDUCHERRY",
    ("DELHI", "NCTOFDELHI"): "explicit state alias DELHI/NCT OF DELHI",
    ("DELHI", "DELHIANDNCR"): "explicit state alias DELHI/DELHI & NCR",
    ("ANDAMANANDNICOBARISLANDS", "ANDAMANANDNICOBARISLAND"): (
        "explicit state alias ANDAMAN AND NICOBAR ISLANDS/ISLAND"
    ),
    ("DADRAANDNAGARHAVELI", "DADARAANDNAGARHAVELLI"): (
        "explicit boundary spelling alias DADRA AND NAGAR HAVELI"
    ),
    ("ARUNACHALPRADESH", "ARUNANCHALPRADESH"): "explicit boundary spelling alias ARUNACHAL PRADESH",
}

HISTORICAL_STATE_CANDIDATES = {
    "TELANGANA": [("ANDHRAPRADESH", "Telangana districts appear under Andhra Pradesh in this boundary layer")],
}

DISTRICT_ALIASES = {
    ("SASNAGAR", "SASNAGAR"): "S.A.S abbreviation punctuation variant",
    ("SAHIBZADAAJITSINGHNAGAR", "SASNAGAR"): "Sahibzada Ajit Singh Nagar / S.A.S Nagar alias",
}


@dataclass
class StateMatch:
    crop_state_name: str
    boundary_state_name: str
    census_version: str
    match_method: str
    match_status: str
    evidence: str


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("&", " AND ")
    text = re.sub(r"(?<=\b[A-Z])\.(?=[A-Z]\b)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text.strip())
    return text.upper()


def compare_key(value: object) -> str:
    normalized = normalize_name(value)
    return re.sub(r"[^A-Z0-9]", "", normalized)


def tokens(value: object) -> set[str]:
    return set(re.findall(r"[A-Z0-9]+", normalize_name(value)))


def token_similarity(left: object, right: object) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def edit_similarity(left: object, right: object) -> float:
    left_key = compare_key(left)
    right_key = compare_key(right)
    if not left_key and not right_key:
        return 1.0
    return difflib.SequenceMatcher(None, left_key, right_key).ratio()


def is_state_alias(crop_state_name: str, boundary_state_name: str) -> str | None:
    pair = (compare_key(crop_state_name), compare_key(boundary_state_name))
    return STATE_ALIASES.get(pair)


def is_district_alias(canonical_district_name: str, boundary_district_name: str) -> str | None:
    left = compare_key(canonical_district_name)
    right = compare_key(boundary_district_name)
    return DISTRICT_ALIASES.get((left, right)) or DISTRICT_ALIASES.get((right, left))


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage1 = pd.read_csv(STAGE1_PATH, keep_default_na=False)
    overrides = pd.read_csv(MANUAL_OVERRIDES_PATH, keep_default_na=False)
    inventory = pd.read_csv(BOUNDARY_INVENTORY_PATH, keep_default_na=False)
    layer_audit = pd.read_csv(LAYER_AUDIT_PATH, keep_default_na=False)
    if not SOURCE_MANIFEST_PATH.exists():
        raise ValueError(f"Missing source manifest: {SOURCE_MANIFEST_PATH}")

    require_columns(
        stage1,
        [
            "district_id",
            "raw_state_name",
            "raw_district_name",
            "canonical_state_name",
            "canonical_district_name",
            "latitude",
            "longitude",
            "override_type",
        ],
        "stage1 crosswalk",
    )
    require_columns(
        inventory,
        [
            "census_version",
            "source_feature_index",
            "raw_state_name",
            "raw_district_name",
            "raw_state_code",
            "raw_district_code",
        ],
        "boundary inventory",
    )
    require_columns(layer_audit, ["census_version", "feature_count"], "boundary layer audit")
    require_columns(overrides, ["raw_state_name", "raw_district_name", "canonical_state_name", "canonical_district_name"], "manual overrides")
    return stage1, overrides, inventory, layer_audit


def validate_inputs(stage1: pd.DataFrame, inventory: pd.DataFrame, layer_audit: pd.DataFrame) -> None:
    if len(stage1) != EXPECTED_STAGE1_ROWS:
        raise ValueError(f"stage1 crosswalk must have {EXPECTED_STAGE1_ROWS} rows, found {len(stage1)}")
    if not stage1["district_id"].is_unique:
        raise ValueError("stage1 district_id is not unique")
    if stage1["latitude"].astype(str).str.strip().ne("").any() or stage1["longitude"].astype(str).str.strip().ne("").any():
        raise ValueError("Stage1 has latitude or longitude filled before boundary matching")
    for census_version, expected_count in EXPECTED_FEATURE_COUNTS.items():
        actual_count = int(layer_audit.loc[layer_audit["census_version"].eq(census_version), "feature_count"].astype(int).sum())
        if actual_count != expected_count:
            raise ValueError(f"{census_version} must have {expected_count} features, found {actual_count}")
        inventory_count = len(inventory[inventory["census_version"].eq(census_version)])
        if inventory_count != expected_count:
            raise ValueError(f"{census_version} inventory must have {expected_count} rows, found {inventory_count}")


def boundary_state_table(inventory: pd.DataFrame, census_version: str) -> pd.DataFrame:
    subset = inventory[inventory["census_version"].eq(census_version)].copy()
    subset["boundary_state_key"] = subset["raw_state_name"].map(compare_key)
    return subset[["raw_state_name", "boundary_state_key"]].drop_duplicates().rename(
        columns={"raw_state_name": "boundary_state_name"}
    )


def build_state_crosswalk(stage1: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[StateMatch] = []
    crop_states = sorted(stage1["canonical_state_name"].drop_duplicates())
    for census_version in CENSUS_VERSIONS:
        states = boundary_state_table(inventory, census_version)
        for crop_state in crop_states:
            crop_key = compare_key(crop_state)
            exact = states[states["boundary_state_key"].eq(crop_key)]
            if len(exact) == 1:
                boundary_state = exact.iloc[0]["boundary_state_name"]
                rows.append(
                    StateMatch(
                        crop_state,
                        boundary_state,
                        census_version,
                        "unique_exact_normalized",
                        "confirmed_by_name",
                        f"Unique normalized state key `{crop_key}`",
                    )
                )
                continue
            alias_matches = []
            for state in states.itertuples(index=False):
                evidence = is_state_alias(crop_state, state.boundary_state_name)
                if evidence:
                    alias_matches.append((state.boundary_state_name, evidence))
            if len(alias_matches) == 1:
                rows.append(
                    StateMatch(
                        crop_state,
                        alias_matches[0][0],
                        census_version,
                        "explicit_state_alias",
                        "confirmed_by_name",
                        alias_matches[0][1],
                    )
                )
                continue
            historical_matches = []
            for historical_key, evidence in HISTORICAL_STATE_CANDIDATES.get(crop_key, []):
                matches = states[states["boundary_state_key"].eq(historical_key)]
                historical_matches.extend((row.boundary_state_name, evidence) for row in matches.itertuples(index=False))
            if historical_matches:
                rows.append(
                    StateMatch(
                        crop_state,
                        historical_matches[0][0],
                        census_version,
                        "cross_state_historical_candidate",
                        "cross_state_historical_candidate",
                        historical_matches[0][1],
                    )
                )
            else:
                rows.append(
                    StateMatch(
                        crop_state,
                        "",
                        census_version,
                        "unmatched",
                        "unresolved",
                        "No unique exact normalized state match or explicit alias",
                    )
                )

    state_crosswalk = pd.DataFrame([row.__dict__ for row in rows])
    confirmed = state_crosswalk[state_crosswalk["match_status"].eq("confirmed_by_name")]
    ambiguous = confirmed.duplicated(["crop_state_name", "census_version"], keep=False)
    if ambiguous.any():
        raise ValueError("State crosswalk contains ambiguous confirmed matches")
    return state_crosswalk


def add_boundary_keys(inventory: pd.DataFrame) -> pd.DataFrame:
    out = inventory.copy()
    out["boundary_state_key"] = out["raw_state_name"].map(compare_key)
    out["boundary_district_key"] = out["raw_district_name"].map(compare_key)
    return out


def candidate_record(
    row: pd.Series,
    census_version: str,
    boundary: pd.Series,
    state_method: str,
    district_method: str,
    rank: int,
    candidate_status: str,
    evidence: str,
) -> dict[str, object]:
    normalized_exact = compare_key(row["canonical_district_name"]) == compare_key(boundary["raw_district_name"])
    return {
        "district_id": row["district_id"],
        "raw_state_name": row["raw_state_name"],
        "raw_district_name": row["raw_district_name"],
        "canonical_state_name": row["canonical_state_name"],
        "canonical_district_name": row["canonical_district_name"],
        "census_version": census_version,
        "boundary_feature_index": boundary["source_feature_index"],
        "boundary_state_name": boundary["raw_state_name"],
        "boundary_district_name": boundary["raw_district_name"],
        "boundary_state_code": boundary["raw_state_code"],
        "boundary_district_code": boundary["raw_district_code"],
        "state_match_method": state_method,
        "district_match_method": district_method,
        "normalized_exact": normalized_exact,
        "string_similarity": round(token_similarity(row["canonical_district_name"], boundary["raw_district_name"]), 6),
        "edit_similarity": round(edit_similarity(row["canonical_district_name"], boundary["raw_district_name"]), 6),
        "candidate_rank": rank,
        "candidate_status": candidate_status,
        "evidence": evidence,
    }


def blank_stage2_row(row: pd.Series, census_version: str, status: str, confidence: str, candidate_count: int, notes: str) -> dict[str, object]:
    return {
        "district_id": row["district_id"],
        "raw_state_name": row["raw_state_name"],
        "raw_district_name": row["raw_district_name"],
        "canonical_state_name": row["canonical_state_name"],
        "canonical_district_name": row["canonical_district_name"],
        "census_version": census_version,
        "boundary_feature_index": "",
        "boundary_state_name": "",
        "boundary_district_name": "",
        "boundary_state_code": "",
        "boundary_district_code": "",
        "match_method": status,
        "match_status": status,
        "match_confidence": confidence,
        "candidate_count": candidate_count,
        "manual_override_used": bool(str(row.get("override_type", "")).strip()),
        "requires_manual_review": status not in MATCHED_STATUSES,
        "notes": notes,
    }


def matched_stage2_row(
    row: pd.Series,
    census_version: str,
    boundary: pd.Series,
    method: str,
    status: str,
    candidate_count: int,
    notes: str,
) -> dict[str, object]:
    return {
        "district_id": row["district_id"],
        "raw_state_name": row["raw_state_name"],
        "raw_district_name": row["raw_district_name"],
        "canonical_state_name": row["canonical_state_name"],
        "canonical_district_name": row["canonical_district_name"],
        "census_version": census_version,
        "boundary_feature_index": boundary["source_feature_index"],
        "boundary_state_name": boundary["raw_state_name"],
        "boundary_district_name": boundary["raw_district_name"],
        "boundary_state_code": boundary["raw_state_code"],
        "boundary_district_code": boundary["raw_district_code"],
        "match_method": method,
        "match_status": status,
        "match_confidence": "confirmed_by_name",
        "candidate_count": candidate_count,
        "manual_override_used": bool(str(row.get("override_type", "")).strip()),
        "requires_manual_review": False,
        "notes": notes,
    }


def top_fuzzy_candidates(row: pd.Series, boundary_subset: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if boundary_subset.empty:
        return boundary_subset.copy()
    candidates = boundary_subset.copy()
    candidates["string_similarity"] = candidates["raw_district_name"].map(
        lambda value: token_similarity(row["canonical_district_name"], value)
    )
    candidates["edit_similarity"] = candidates["raw_district_name"].map(
        lambda value: edit_similarity(row["canonical_district_name"], value)
    )
    candidates["score"] = candidates[["string_similarity", "edit_similarity"]].mean(axis=1)
    candidates = candidates.sort_values(["score", "edit_similarity", "string_similarity", "raw_district_name"], ascending=[False, False, False, True])
    return candidates.head(limit)


def match_one(row: pd.Series, census_version: str, state_row: pd.Series, inventory: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates_out: list[dict[str, object]] = []
    manual_override_used = bool(str(row.get("override_type", "")).strip())
    canonical_key = compare_key(row["canonical_district_name"])
    state_status = state_row["match_status"]
    boundary_state_name = state_row["boundary_state_name"]
    state_method = state_row["match_method"]
    notes_prefix = state_row["evidence"]

    if state_status == "confirmed_by_name":
        boundary_subset = inventory[
            inventory["census_version"].eq(census_version) & inventory["raw_state_name"].eq(boundary_state_name)
        ].copy()
        exact = boundary_subset[boundary_subset["boundary_district_key"].eq(canonical_key)]
        if len(exact) > 1:
            raise ValueError(
                f"Exact district match has more than one candidate for {row['district_id']} {census_version}: "
                f"{row['canonical_state_name']} / {row['canonical_district_name']}"
            )
        if len(exact) == 1:
            boundary = exact.iloc[0]
            status = "matched_manual_override" if manual_override_used else "matched_exact_unique"
            method = "manual_override_strong" if manual_override_used else "exact_unique"
            evidence = (
                "Manual override canonical name uniquely matched boundary district"
                if manual_override_used
                else "Unique exact normalized district key within confirmed boundary state"
            )
            candidates_out.append(candidate_record(row, census_version, boundary, state_method, method, 1, status, evidence))
            return matched_stage2_row(row, census_version, boundary, method, status, 1, evidence), candidates_out

        alias_rows = []
        for boundary in boundary_subset.itertuples(index=False):
            evidence = is_district_alias(row["canonical_district_name"], boundary.raw_district_name)
            if evidence:
                alias_rows.append((pd.Series(boundary._asdict()), evidence))
        if len(alias_rows) == 1:
            boundary, evidence = alias_rows[0]
            status = "matched_manual_override" if manual_override_used else "matched_alias_unique"
            method = "manual_override_strong" if manual_override_used else "alias_unique_reviewed"
            evidence = (
                f"Manual override canonical name uniquely matched boundary alias: {evidence}"
                if manual_override_used
                else evidence
            )
            candidates_out.append(candidate_record(row, census_version, boundary, state_method, method, 1, status, evidence))
            return matched_stage2_row(row, census_version, boundary, method, status, 1, evidence), candidates_out
        if len(alias_rows) > 1:
            raise ValueError(f"Alias district match has more than one candidate for {row['district_id']} {census_version}")

        fuzzy = top_fuzzy_candidates(row, boundary_subset)
        for rank, (_, boundary) in enumerate(fuzzy.iterrows(), start=1):
            evidence = "Fuzzy candidate only within confirmed boundary state; not auto-confirmed"
            candidates_out.append(
                candidate_record(row, census_version, boundary, state_method, "fuzzy_candidate", rank, "fuzzy_candidate_only", evidence)
            )
        status = "fuzzy_candidate_only" if len(fuzzy) else "unmatched"
        return (
            blank_stage2_row(
                row,
                census_version,
                status,
                "unresolved",
                len(fuzzy),
                f"{notes_prefix}; no exact or explicit alias district match",
            ),
            candidates_out,
        )

    if state_status == "cross_state_historical_candidate":
        boundary_subset = inventory[
            inventory["census_version"].eq(census_version) & inventory["raw_state_name"].eq(boundary_state_name)
        ].copy()
        exact = boundary_subset[boundary_subset["boundary_district_key"].eq(canonical_key)]
        source = exact if not exact.empty else top_fuzzy_candidates(row, boundary_subset)
        for rank, (_, boundary) in enumerate(source.head(5).iterrows(), start=1):
            method = "cross_state_exact_candidate" if compare_key(boundary["raw_district_name"]) == canonical_key else "cross_state_fuzzy_candidate"
            evidence = f"{notes_prefix}; cross-state historical candidate only, not confirmed"
            candidates_out.append(
                candidate_record(
                    row,
                    census_version,
                    boundary,
                    state_method,
                    method,
                    rank,
                    "cross_state_historical_candidate",
                    evidence,
                )
            )
        confidence = "strong_candidate" if not exact.empty else "unresolved"
        return (
            blank_stage2_row(
                row,
                census_version,
                "cross_state_historical_candidate" if len(source) else "unmatched",
                confidence,
                len(source.head(5)),
                f"{notes_prefix}; boundary state is not confirmed for crop state",
            ),
            candidates_out,
        )

    return blank_stage2_row(row, census_version, "unmatched", "unresolved", 0, notes_prefix), candidates_out


def build_matches(stage1: pd.DataFrame, inventory: pd.DataFrame, state_crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = add_boundary_keys(inventory)
    stage_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for _, row in stage1.iterrows():
        for census_version in CENSUS_VERSIONS:
            state_match = state_crosswalk[
                state_crosswalk["crop_state_name"].eq(row["canonical_state_name"])
                & state_crosswalk["census_version"].eq(census_version)
            ]
            if len(state_match) != 1:
                raise ValueError(f"Expected one state crosswalk row for {row['canonical_state_name']} {census_version}")
            stage_row, candidates = match_one(row, census_version, state_match.iloc[0], inventory)
            stage_rows.append(stage_row)
            candidate_rows.extend(candidates)
    return pd.DataFrame(stage_rows), pd.DataFrame(candidate_rows)


def build_status(stage2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for district_id, group in stage2.groupby("district_id", sort=True):
        if len(group) != 2:
            raise ValueError(f"Expected two stage2 rows for {district_id}, found {len(group)}")
        row_2001 = group[group["census_version"].eq("Census 2001")].iloc[0]
        row_2011 = group[group["census_version"].eq("Census 2011")].iloc[0]
        matched_2001 = row_2001["match_status"] in MATCHED_STATUSES
        matched_2011 = row_2011["match_status"] in MATCHED_STATUSES
        rows.append(
            {
                "district_id": district_id,
                "match_2001_status": row_2001["match_status"],
                "match_2011_status": row_2011["match_status"],
                "matched_in_both": matched_2001 and matched_2011,
                "matched_in_2001_only": matched_2001 and not matched_2011,
                "matched_in_2011_only": matched_2011 and not matched_2001,
                "unresolved_in_both": not matched_2001 and not matched_2011,
                "requires_manual_review": bool(row_2001["requires_manual_review"]) or bool(row_2011["requires_manual_review"]),
            }
        )
    status = pd.DataFrame(rows)
    if len(status) != EXPECTED_STAGE1_ROWS:
        raise ValueError(f"Status report must have {EXPECTED_STAGE1_ROWS} rows, found {len(status)}")
    return status


def make_summary(stage2: pd.DataFrame, status: pd.DataFrame, state_crosswalk: pd.DataFrame) -> str:
    lines = [
        "# District Boundary Matching Summary",
        "",
        "Crop districts were compared to DataMeet Census 2001 and Census 2011 boundary-name inventories.",
        "Fuzzy matching was used only to generate candidates. No representative points or coordinates were created.",
        "",
    ]
    for census_version in CENSUS_VERSIONS:
        subset = stage2[stage2["census_version"].eq(census_version)]
        counts = subset["match_status"].value_counts()
        lines.extend(
            [
                f"## {census_version}",
                "",
                f"- Crop districts: {len(subset)}",
                f"- Exact unique matches: {int(counts.get('matched_exact_unique', 0))}",
                f"- Alias matches: {int(counts.get('matched_alias_unique', 0))}",
                f"- Manual override matches: {int(counts.get('matched_manual_override', 0))}",
                f"- Multiple candidates: {int(counts.get('multiple_candidates', 0))}",
                f"- Fuzzy-only candidates: {int(counts.get('fuzzy_candidate_only', 0))}",
                f"- Cross-state candidates: {int(counts.get('cross_state_historical_candidate', 0))}",
                f"- Unmatched: {int(counts.get('unmatched', 0))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-Layer Status",
            "",
            f"- Matched in both layers: {int(status['matched_in_both'].sum())}",
            f"- Matched only in 2001: {int(status['matched_in_2001_only'].sum())}",
            f"- Matched only in 2011: {int(status['matched_in_2011_only'].sum())}",
            f"- Unresolved in both layers: {int(status['unresolved_in_both'].sum())}",
            "",
            "## Unresolved Districts",
            "",
        ]
    )
    unresolved = stage2[~stage2["match_status"].isin(MATCHED_STATUSES)].sort_values(
        ["census_version", "canonical_state_name", "canonical_district_name"]
    )
    if unresolved.empty:
        lines.append("- None.")
    else:
        for row in unresolved.itertuples(index=False):
            lines.append(
                f"- `{row.census_version}` `{row.raw_state_name}` / `{row.raw_district_name}` "
                f"as `{row.canonical_state_name}` / `{row.canonical_district_name}`: `{row.match_status}`"
            )

    punjab = stage2[stage2["raw_state_name"].eq("Punjab") & stage2["raw_district_name"].eq("S")]
    telangana_states = state_crosswalk[state_crosswalk["crop_state_name"].eq("Telangana")]
    telangana = stage2[stage2["canonical_state_name"].eq("Telangana")]
    lines.extend(["", "## Punjab / S", ""])
    for row in punjab.itertuples(index=False):
        lines.append(
            f"- `{row.census_version}`: `{row.match_status}` against `{row.boundary_state_name}` / "
            f"`{row.boundary_district_name}`; method `{row.match_method}`"
        )
    lines.extend(["", "## Telangana", ""])
    for row in telangana_states.itertuples(index=False):
        lines.append(
            f"- State crosswalk `{row.census_version}`: `{row.match_status}` to `{row.boundary_state_name}`; "
            f"{row.evidence}"
        )
    telangana_counts = telangana.groupby(["census_version", "match_status"]).size().reset_index(name="count")
    for row in telangana_counts.itertuples(index=False):
        lines.append(f"- District rows `{row.census_version}` `{row.match_status}`: {row.count}")
    lines.extend(["", "Coordinates assigned: no.", ""])
    return "\n".join(lines)


def validate_outputs(stage1: pd.DataFrame, stage2: pd.DataFrame, status: pd.DataFrame, state_crosswalk: pd.DataFrame) -> None:
    if len(stage2) != EXPECTED_STAGE1_ROWS * 2:
        raise ValueError(f"Stage2 must have {EXPECTED_STAGE1_ROWS * 2} rows, found {len(stage2)}")
    if set(stage1["district_id"]) != set(stage2["district_id"]):
        raise ValueError("Stage2 lost at least one district_id")
    if len(status) != EXPECTED_STAGE1_ROWS:
        raise ValueError(f"Status report must have {EXPECTED_STAGE1_ROWS} rows, found {len(status)}")
    bad_status = sorted(set(stage2["match_status"]) - ALLOWED_MATCH_STATUSES)
    bad_confidence = sorted(set(stage2["match_confidence"]) - ALLOWED_CONFIDENCE)
    if bad_status:
        raise ValueError(f"Invalid match_status values: {bad_status}")
    if bad_confidence:
        raise ValueError(f"Invalid match_confidence values: {bad_confidence}")
    confirmed = state_crosswalk[state_crosswalk["match_status"].eq("confirmed_by_name")]
    if confirmed.duplicated(["crop_state_name", "census_version"]).any():
        raise ValueError("State crosswalk contains ambiguous confirmed match")
    if "latitude" in stage2.columns or "longitude" in stage2.columns:
        raise ValueError("Stage2 unexpectedly contains latitude or longitude columns")
    result = subprocess.run(
        ["git", "ls-files", "data/external/datameet_districts/"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("External shapefiles are tracked by Git")


def write_outputs(state_crosswalk: pd.DataFrame, stage2: pd.DataFrame, candidates: pd.DataFrame, status: pd.DataFrame) -> None:
    STATE_CROSSWALK_PATH.parent.mkdir(parents=True, exist_ok=True)
    state_crosswalk.to_csv(STATE_CROSSWALK_PATH, index=False, lineterminator="\n")
    stage2.to_csv(STAGE2_PATH, index=False, lineterminator="\n")
    candidates.to_csv(CANDIDATES_PATH, index=False, lineterminator="\n")
    status.to_csv(STATUS_PATH, index=False, lineterminator="\n")
    SUMMARY_PATH.write_text(make_summary(stage2, status, state_crosswalk), encoding="utf-8", newline="\n")


def main() -> int:
    stage1, _overrides, inventory, layer_audit = load_inputs()
    validate_inputs(stage1, inventory, layer_audit)
    state_crosswalk = build_state_crosswalk(stage1, inventory)
    stage2, candidates = build_matches(stage1, inventory, state_crosswalk)
    status = build_status(stage2)
    validate_outputs(stage1, stage2, status, state_crosswalk)
    write_outputs(state_crosswalk, stage2, candidates, status)

    for census_version in CENSUS_VERSIONS:
        counts = stage2[stage2["census_version"].eq(census_version)]["match_status"].value_counts()
        print(f"{census_version} status counts:")
        for status_name in sorted(ALLOWED_MATCH_STATUSES):
            print(f"  {status_name}={int(counts.get(status_name, 0))}")
    print(f"matched_in_both={int(status['matched_in_both'].sum())}")
    print(f"matched_in_2001_only={int(status['matched_in_2001_only'].sum())}")
    print(f"matched_in_2011_only={int(status['matched_in_2011_only'].sum())}")
    print(f"unresolved_in_both={int(status['unresolved_in_both'].sum())}")
    print("coordinates_assigned=false")
    print("external_files_tracked=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
