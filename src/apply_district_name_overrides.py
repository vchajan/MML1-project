from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "data" / "reference" / "district_crosswalk_template_1997_2014.csv"
OVERRIDES_PATH = REPO_ROOT / "data" / "reference" / "district_name_manual_overrides.csv"
RAW_CROP_PATH = REPO_ROOT / "data" / "raw" / "Indian_crop_production_yield_dataset.csv"
AUDIT_PATH = REPO_ROOT / "reports" / "district_name_audit.csv"
ANOMALY_DETAILS_PATH = REPO_ROOT / "reports" / "district_name_anomaly_details.csv"

STAGE1_PATH = REPO_ROOT / "data" / "reference" / "district_crosswalk_stage1.csv"
APPLICATION_REPORT_PATH = REPO_ROOT / "reports" / "district_override_application_report.csv"
APPLICATION_SUMMARY_PATH = REPO_ROOT / "reports" / "district_override_application_summary.md"

EXPECTED_ROWS = 727
TEMPLATE_REQUIRED_COLUMNS = [
    "district_id",
    "raw_state_name",
    "raw_district_name",
    "canonical_state_name",
    "canonical_district_name",
    "valid_from",
    "valid_to",
    "latitude",
    "longitude",
    "point_method",
    "match_method",
    "match_status",
    "geometry_source",
    "coordinate_source",
    "crop_rows",
    "unique_years",
    "unique_crops",
    "unique_seasons",
    "notes",
]
OVERRIDE_REQUIRED_COLUMNS = [
    "raw_state_name",
    "raw_district_name",
    "canonical_state_name",
    "canonical_district_name",
    "override_type",
    "evidence",
    "confidence",
    "review_status",
]
STAGE1_COLUMNS = [
    "district_id",
    "raw_state_name",
    "raw_district_name",
    "canonical_state_name",
    "canonical_district_name",
    "valid_from",
    "valid_to",
    "latitude",
    "longitude",
    "point_method",
    "match_method",
    "match_status",
    "override_type",
    "override_confidence",
    "override_evidence",
    "override_review_status",
    "geometry_source",
    "coordinate_source",
    "crop_rows",
    "unique_years",
    "unique_crops",
    "unique_seasons",
    "notes",
]
ALLOWED_CONFIDENCE = {"confirmed", "strong", "unresolved"}
ALLOWED_OVERRIDE_TYPES = {"alias", "truncated_name", "renamed", "data_corruption"}


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(TEMPLATE_PATH, keep_default_na=False)
    overrides = pd.read_csv(OVERRIDES_PATH, keep_default_na=False)
    audit = pd.read_csv(AUDIT_PATH, keep_default_na=False)
    anomaly_details = pd.read_csv(ANOMALY_DETAILS_PATH, keep_default_na=False)

    require_columns(template, TEMPLATE_REQUIRED_COLUMNS, "district crosswalk template")
    require_columns(overrides, OVERRIDE_REQUIRED_COLUMNS, "manual overrides")
    require_columns(audit, ["district_id"], "district name audit")
    require_columns(anomaly_details, ["district_id"], "district anomaly details")

    if len(template) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} template rows, found {len(template)}")
    if not template["district_id"].is_unique:
        raise ValueError("Template district_id values are not unique")

    override_duplicates = overrides.duplicated(["raw_state_name", "raw_district_name"], keep=False)
    if override_duplicates.any():
        duplicate_rows = overrides.loc[override_duplicates, ["raw_state_name", "raw_district_name"]]
        raise ValueError(
            "Override raw_state_name + raw_district_name is not unique: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    bad_types = sorted(set(overrides["override_type"]) - ALLOWED_OVERRIDE_TYPES)
    bad_confidence = sorted(set(overrides["confidence"]) - ALLOWED_CONFIDENCE)
    if bad_types:
        raise ValueError(f"Invalid override_type values: {bad_types}")
    if bad_confidence:
        raise ValueError(f"Invalid confidence values: {bad_confidence}")
    if overrides["evidence"].astype(str).str.strip().eq("").any():
        raise ValueError("Every override row must include evidence")

    return template, overrides, audit, anomaly_details


def build_stage1(template: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage = template.copy()
    original_raw_names = stage[["district_id", "raw_state_name", "raw_district_name"]].copy()

    stage["canonical_state_name"] = stage["raw_state_name"]
    stage["canonical_district_name"] = stage["raw_district_name"]
    stage["match_method"] = "exact_pending_boundary_check"
    stage["match_status"] = "pending_boundary_match"
    stage["override_type"] = ""
    stage["override_confidence"] = ""
    stage["override_evidence"] = ""
    stage["override_review_status"] = ""

    # Stage 1 is name-only. Coordinates and geographic confirmation remain blank.
    stage["latitude"] = ""
    stage["longitude"] = ""
    stage["point_method"] = ""
    stage["geometry_source"] = ""
    stage["coordinate_source"] = ""

    report_rows: list[dict[str, object]] = []
    applied_indices: list[int] = []

    for override in overrides.itertuples(index=False):
        mask = stage["raw_state_name"].eq(override.raw_state_name) & stage["raw_district_name"].eq(
            override.raw_district_name
        )
        matches = stage.loc[mask]
        if matches.empty:
            raise ValueError(
                "Override references missing raw name: "
                f"{override.raw_state_name} / {override.raw_district_name}"
            )
        if len(matches) != 1:
            raise ValueError(
                "Override would apply multiple times: "
                f"{override.raw_state_name} / {override.raw_district_name} -> {len(matches)} rows"
            )

        idx = matches.index[0]
        applied_indices.append(int(idx))
        stage.loc[idx, "canonical_state_name"] = override.canonical_state_name
        stage.loc[idx, "canonical_district_name"] = override.canonical_district_name
        stage.loc[idx, "override_type"] = override.override_type
        stage.loc[idx, "override_confidence"] = override.confidence
        stage.loc[idx, "override_evidence"] = override.evidence
        stage.loc[idx, "override_review_status"] = override.review_status

        if override.raw_state_name == "Punjab" and override.raw_district_name == "S":
            stage.loc[idx, "match_method"] = "manual_override_strong"
            stage.loc[idx, "match_status"] = "pending_boundary_confirmation"
        else:
            method_suffix = str(override.confidence).strip() or "reviewed"
            stage.loc[idx, "match_method"] = f"manual_override_{method_suffix}"
            stage.loc[idx, "match_status"] = "pending_boundary_confirmation"

        report_rows.append(
            {
                "district_id": stage.loc[idx, "district_id"],
                "raw_state_name": override.raw_state_name,
                "raw_district_name": override.raw_district_name,
                "canonical_state_name": override.canonical_state_name,
                "canonical_district_name": override.canonical_district_name,
                "override_type": override.override_type,
                "override_confidence": override.confidence,
                "override_review_status": override.review_status,
                "match_method": stage.loc[idx, "match_method"],
                "match_status": stage.loc[idx, "match_status"],
                "override_evidence": override.evidence,
            }
        )

    if len(applied_indices) != len(set(applied_indices)):
        raise ValueError("At least one template row received more than one override")

    raw_after = stage[["district_id", "raw_state_name", "raw_district_name"]].copy()
    if not original_raw_names.equals(raw_after):
        raise ValueError("Raw state or district names changed during override application")

    stage = stage[STAGE1_COLUMNS]
    report = pd.DataFrame(
        report_rows,
        columns=[
            "district_id",
            "raw_state_name",
            "raw_district_name",
            "canonical_state_name",
            "canonical_district_name",
            "override_type",
            "override_confidence",
            "override_review_status",
            "match_method",
            "match_status",
            "override_evidence",
        ],
    )
    return stage, report


def validate_stage1(template: pd.DataFrame, overrides: pd.DataFrame, stage: pd.DataFrame) -> None:
    if len(stage) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} output rows, found {len(stage)}")
    if not stage["district_id"].is_unique:
        raise ValueError("Output district_id values are not unique")

    if not template[["raw_state_name", "raw_district_name"]].equals(
        stage[["raw_state_name", "raw_district_name"]]
    ):
        raise ValueError("Raw district names changed in output")

    applied = stage["override_type"].astype(str).str.strip().ne("")
    if int(applied.sum()) != len(overrides):
        raise ValueError(f"Expected {len(overrides)} applied overrides, found {int(applied.sum())}")

    if stage["latitude"].astype(str).str.strip().ne("").any() or stage["longitude"].astype(str).str.strip().ne("").any():
        raise ValueError("Latitude or longitude was unexpectedly filled")

    disallowed_confirmed = stage["match_status"].astype(str).str.contains("confirmed|matched", case=False, regex=True)
    if disallowed_confirmed.any():
        raise ValueError("A row was marked as geographically confirmed or matched")

    punjab = stage[stage["raw_state_name"].eq("Punjab") & stage["raw_district_name"].eq("S")]
    if len(punjab) != 1:
        raise ValueError("Expected exactly one Punjab / S row in stage1 crosswalk")
    punjab_row = punjab.iloc[0]
    if punjab_row["canonical_state_name"] != "Punjab" or punjab_row["canonical_district_name"] != "S.A.S NAGAR":
        raise ValueError("Punjab / S override did not set the expected canonical district")
    if punjab_row["match_method"] != "manual_override_strong":
        raise ValueError("Punjab / S override did not set match_method=manual_override_strong")
    if punjab_row["match_status"] != "pending_boundary_confirmation":
        raise ValueError("Punjab / S override did not set match_status=pending_boundary_confirmation")


def make_summary(
    stage: pd.DataFrame,
    report: pd.DataFrame,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> str:
    applied_count = len(report)
    unchanged_count = int(
        (
            stage["canonical_state_name"].eq(stage["raw_state_name"])
            & stage["canonical_district_name"].eq(stage["raw_district_name"])
        ).sum()
    )
    pending_boundary_match = int(stage["match_status"].eq("pending_boundary_match").sum())
    pending_boundary_confirmation = int(stage["match_status"].eq("pending_boundary_confirmation").sum())
    coordinates_blank = bool(
        stage["latitude"].astype(str).str.strip().eq("").all()
        and stage["longitude"].astype(str).str.strip().eq("").all()
    )
    protected_unchanged = hashes_before == hashes_after

    lines = [
        "# District Override Application Summary",
        "",
        "District name overrides were applied to a stage-1 crosswalk only.",
        "No maps were downloaded, no geocoding was performed, and no coordinates were assigned.",
        "",
        "## Counts",
        "",
        f"- Input districts: {len(stage)}",
        f"- Applied override rules: {applied_count}",
        f"- Unchanged canonical names: {unchanged_count}",
        f"- `pending_boundary_match`: {pending_boundary_match}",
        f"- `pending_boundary_confirmation`: {pending_boundary_confirmation}",
        "",
        "## Override Rules",
        "",
    ]
    if report.empty:
        lines.append("- None.")
    else:
        for row in report.itertuples(index=False):
            lines.append(
                f"- `{row.raw_state_name}` / `{row.raw_district_name}` -> "
                f"`{row.canonical_state_name}` / `{row.canonical_district_name}`; "
                f"type `{row.override_type}`; confidence `{row.override_confidence}`; "
                f"status `{row.match_status}`"
            )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Coordinates added: {'no' if coordinates_blank else 'yes'}",
            f"- Template and raw crop data unchanged: {'yes' if protected_unchanged else 'no'}",
            f"- Template SHA-256 before/after: `{hashes_before['template']}` / `{hashes_after['template']}`",
            f"- Raw crop SHA-256 before/after: `{hashes_before['raw_crop']}` / `{hashes_after['raw_crop']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    hashes_before = {
        "template": file_sha256(TEMPLATE_PATH),
        "raw_crop": file_sha256(RAW_CROP_PATH),
    }
    template, overrides, _audit, _anomaly_details = load_inputs()
    stage, report = build_stage1(template, overrides)
    validate_stage1(template, overrides, stage)

    hashes_after = {
        "template": file_sha256(TEMPLATE_PATH),
        "raw_crop": file_sha256(RAW_CROP_PATH),
    }
    if hashes_before != hashes_after:
        raise ValueError("Template or raw crop dataset changed during override application")

    STAGE1_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPLICATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage.to_csv(STAGE1_PATH, index=False, lineterminator="\n")
    report.to_csv(APPLICATION_REPORT_PATH, index=False, lineterminator="\n")
    APPLICATION_SUMMARY_PATH.write_text(
        make_summary(stage, report, hashes_before, hashes_after),
        encoding="utf-8",
        newline="\n",
    )

    pending_boundary_match = int(stage["match_status"].eq("pending_boundary_match").sum())
    pending_boundary_confirmation = int(stage["match_status"].eq("pending_boundary_confirmation").sum())
    punjab = stage[stage["raw_state_name"].eq("Punjab") & stage["raw_district_name"].eq("S")].iloc[0]

    print("District name overrides applied")
    print(f"stage1_rows={len(stage)}")
    print(f"applied_overrides={len(report)}")
    print(
        "punjab_s="
        f"{punjab['raw_state_name']} / {punjab['raw_district_name']} -> "
        f"{punjab['canonical_state_name']} / {punjab['canonical_district_name']} "
        f"({punjab['match_method']}, {punjab['match_status']})"
    )
    print(f"pending_boundary_match={pending_boundary_match}")
    print(f"pending_boundary_confirmation={pending_boundary_confirmation}")
    print("coordinates_added=false")
    print("template_modified=false")
    print("raw_crop_modified=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"District override application failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
