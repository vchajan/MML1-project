from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "reference" / "required_districts_1997_2014.csv"
OUTPUT_CSV = REPO_ROOT / "reports" / "district_name_audit.csv"
OUTPUT_SUMMARY = REPO_ROOT / "reports" / "district_name_audit_summary.md"
EXPECTED_ROWS = 727

REQUIRED_COLUMNS = [
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

DASH_CHARS = {
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
    "\u2212",
}
APOSTROPHE_CHARS = {"\u2018", "\u2019", "\u201b", "\u2032", "\u00b4", "`"}
ALLOWED_PUNCTUATION = set(" .'-()/&")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_display(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    for char in DASH_CHARS:
        text = text.replace(char, "-")
    for char in APOSTROPHE_CHARS:
        text = text.replace(char, "'")
    text = text.replace("\\", "/")
    text = re.sub(r"\s+", " ", text.strip())
    return text.upper()


def compare_key(normalized_name: str) -> str:
    key = re.sub(r"[.'`\-\(\)\[\]\{\}]", "", normalized_name)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def compact_name_length(normalized_name: str) -> int:
    return len(re.sub(r"[^A-Z0-9]", "", normalized_name))


def contains_unusual_punctuation(value: object) -> int:
    text = "" if pd.isna(value) else unicodedata.normalize("NFKC", str(value))
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(("P", "S")) and char not in ALLOWED_PUNCTUATION:
            return 1
    return 0


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    if not df["district_id"].is_unique:
        duplicates = df.loc[df["district_id"].duplicated(keep=False), "district_id"].tolist()
        raise ValueError(f"district_id is not unique. Duplicates: {duplicates[:10]}")

    combo_duplicates = df.duplicated(["State_Name", "District_Name"], keep=False)
    if combo_duplicates.any():
        duplicate_rows = df.loc[combo_duplicates, ["State_Name", "District_Name"]].head(10)
        raise ValueError(
            "State_Name + District_Name is not unique. "
            f"Examples: {duplicate_rows.to_dict(orient='records')}"
        )


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    audited = df.copy()
    audited["normalized_state_name"] = audited["State_Name"].map(normalize_display)
    audited["normalized_district_name"] = audited["District_Name"].map(normalize_display)
    audited["state_compare_key"] = audited["normalized_state_name"].map(compare_key)
    audited["district_compare_key"] = audited["normalized_district_name"].map(compare_key)

    compact_lengths = audited["normalized_district_name"].map(compact_name_length)
    short_threshold = max(2, int(round(compact_lengths.median() * 0.25)))

    audited["flag_name_too_short"] = compact_lengths.le(2).astype(int)
    audited["flag_single_character_name"] = compact_lengths.eq(1).astype(int)
    audited["flag_contains_digit"] = audited["District_Name"].astype(str).str.contains(r"\d", regex=True).astype(int)
    audited["flag_contains_parentheses"] = audited["District_Name"].astype(str).str.contains(r"[()]", regex=True).astype(int)
    audited["flag_contains_unusual_punctuation"] = audited["District_Name"].map(contains_unusual_punctuation).astype(int)
    audited["flag_multiple_spaces_original"] = audited["District_Name"].astype(str).str.contains(r"\s{2,}", regex=True).astype(int)
    audited["flag_leading_or_trailing_space_original"] = (
        audited["District_Name"].astype(str) != audited["District_Name"].astype(str).str.strip()
    ).astype(int)

    within_state_group_size = audited.groupby(["state_compare_key", "district_compare_key"])["district_id"].transform("size")
    raw_names_per_normalized = audited.groupby(["state_compare_key", "district_compare_key"])["District_Name"].transform(
        "nunique"
    )
    states_per_district_key = audited.groupby("district_compare_key")["state_compare_key"].transform("nunique")

    audited["flag_duplicate_normalized_within_state"] = within_state_group_size.gt(1).astype(int)
    audited["flag_same_district_name_multiple_states"] = states_per_district_key.gt(1).astype(int)
    audited["flag_multiple_raw_names_same_normalized_key"] = raw_names_per_normalized.gt(1).astype(int)

    ends_with_period = audited["normalized_district_name"].str.endswith(".")
    ends_with_short_abbrev = audited["normalized_district_name"].str.contains(r"\b[A-Z]{1,3}\.$", regex=True)
    unusually_short = compact_lengths.le(short_threshold)
    audited["flag_possible_truncation"] = (
        audited["flag_single_character_name"].eq(1)
        | compact_lengths.le(2)
        | ends_with_period
        | ends_with_short_abbrev
        | unusually_short
    ).astype(int)

    audited["issue_count"] = audited[FLAG_COLUMNS].sum(axis=1)

    high_condition = (
        audited["flag_name_too_short"].eq(1)
        | audited["flag_single_character_name"].eq(1)
        | audited["flag_duplicate_normalized_within_state"].eq(1)
        | audited["flag_multiple_raw_names_same_normalized_key"].eq(1)
        | audited["flag_possible_truncation"].eq(1)
    )
    medium_condition = (
        audited["flag_contains_digit"].eq(1)
        | audited["flag_contains_parentheses"].eq(1)
        | audited["flag_contains_unusual_punctuation"].eq(1)
        | audited["flag_same_district_name_multiple_states"].eq(1)
        | audited["flag_multiple_spaces_original"].eq(1)
        | audited["flag_leading_or_trailing_space_original"].eq(1)
    )

    audited["review_priority"] = "low"
    audited.loc[medium_condition, "review_priority"] = "medium"
    audited.loc[high_condition, "review_priority"] = "high"
    audited["flag_requires_manual_review"] = audited["issue_count"].gt(0).astype(int)
    audited["review_status"] = "no_issue_detected"
    audited.loc[audited["flag_requires_manual_review"].eq(1), "review_status"] = "pending"

    return audited


def reason_list(row: pd.Series) -> str:
    return ", ".join(flag for flag in FLAG_COLUMNS if int(row[flag]) == 1)


def format_records(rows: pd.DataFrame, limit: int | None = None) -> list[str]:
    if rows.empty:
        return ["- None."]
    formatted: list[str] = []
    selected = rows if limit is None else rows.head(limit)
    for _, row in selected.iterrows():
        formatted.append(f"- `{row['State_Name']}` / `{row['District_Name']}`: {reason_list(row)}")
    if limit is not None and len(rows) > limit:
        formatted.append(f"- ... {len(rows) - limit} more")
    return formatted


def make_summary(audited: pd.DataFrame, input_hash_before: str, input_hash_after: str) -> str:
    problem_rows = audited[audited["flag_requires_manual_review"].eq(1)]
    high_rows = audited[audited["review_priority"].eq("high")].sort_values(["State_Name", "District_Name"])

    normalized_conflict_groups = (
        audited.groupby(["state_compare_key", "district_compare_key"])
        .agg(
            state_names=("State_Name", lambda values: sorted(set(values))),
            district_names=("District_Name", lambda values: sorted(set(values))),
            row_count=("district_id", "size"),
            raw_name_count=("District_Name", "nunique"),
        )
        .reset_index()
    )
    normalized_conflict_groups = normalized_conflict_groups[
        (normalized_conflict_groups["row_count"] > 1) | (normalized_conflict_groups["raw_name_count"] > 1)
    ]

    multi_state_groups = (
        audited.groupby("district_compare_key")
        .agg(
            states=("State_Name", lambda values: sorted(set(values))),
            district_names=("District_Name", lambda values: sorted(set(values))),
        )
        .reset_index()
    )
    multi_state_groups["state_count"] = multi_state_groups["states"].map(len)
    multi_state_groups = multi_state_groups[multi_state_groups["state_count"] > 1].sort_values("district_compare_key")

    raw_name_groups = normalized_conflict_groups[normalized_conflict_groups["raw_name_count"] > 1].sort_values(
        ["state_compare_key", "district_compare_key"]
    )

    lines: list[str] = [
        "# District Name Audit Summary",
        "",
        "This audit flags historical district-name issues before district crosswalk construction.",
        "No geocoding, map-layer download, coordinate assignment, or manual district-name correction was performed.",
        "",
        "## Counts",
        "",
        f"- Input district rows: {len(audited)}",
        f"- States: {audited['State_Name'].nunique()}",
        f"- Unique district_id values: {audited['district_id'].nunique()}",
        f"- Problematic district rows: {len(problem_rows)}",
        f"- High priority rows: {(audited['review_priority'] == 'high').sum()}",
        f"- Medium priority rows: {(audited['review_priority'] == 'medium').sum()}",
        f"- Low priority rows: {(audited['review_priority'] == 'low').sum()}",
        f"- Normalized conflict groups: {len(normalized_conflict_groups)}",
        "",
        "## High-Priority Names",
        "",
        *format_records(high_rows),
        "",
        "## District Names Appearing In Multiple States",
        "",
    ]

    if multi_state_groups.empty:
        lines.append("- None.")
    else:
        for _, row in multi_state_groups.iterrows():
            states = "; ".join(row["states"])
            names = "; ".join(row["district_names"])
            lines.append(f"- `{row['district_compare_key']}`: names `{names}`; states `{states}`")

    lines.extend(["", "## Raw Names Sharing The Same Normalized Key", ""])
    if raw_name_groups.empty:
        lines.append("- None.")
    else:
        for _, row in raw_name_groups.iterrows():
            states = "; ".join(row["state_names"])
            names = "; ".join(row["district_names"])
            lines.append(
                f"- state key `{row['state_compare_key']}`, district key `{row['district_compare_key']}`: "
                f"raw names `{names}`; states `{states}`"
            )

    lines.extend(
        [
            "",
            "## Input Integrity",
            "",
            f"- Input SHA-256 before audit: `{input_hash_before}`",
            f"- Input SHA-256 after audit: `{input_hash_after}`",
            "- Input file was not modified by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    input_hash_before = file_sha256(INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    validate_input(df)

    audited = add_audit_columns(df)
    if len(audited) != len(df):
        raise ValueError(f"Output row count mismatch: input={len(df)}, output={len(audited)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")

    input_hash_after = file_sha256(INPUT_PATH)
    if input_hash_before != input_hash_after:
        raise ValueError("Input file hash changed during audit")

    summary = make_summary(audited, input_hash_before, input_hash_after)
    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8", newline="\n")

    problem_count = int(audited["flag_requires_manual_review"].sum())
    high_count = int((audited["review_priority"] == "high").sum())
    medium_count = int((audited["review_priority"] == "medium").sum())
    low_count = int((audited["review_priority"] == "low").sum())
    normalized_conflict_count = int(
        (
            audited.groupby(["state_compare_key", "district_compare_key"])["district_id"].transform("size").gt(1)
            | audited["flag_multiple_raw_names_same_normalized_key"].eq(1)
        ).sum()
    )

    print("District-name audit completed")
    print(f"input_rows={len(df)}")
    print(f"states={df['State_Name'].nunique()}")
    print(f"problematic_rows={problem_count}")
    print(f"priority_high={high_count}")
    print(f"priority_medium={medium_count}")
    print(f"priority_low={low_count}")
    print(f"normalized_conflict_rows={normalized_conflict_count}")
    print(f"output_csv={OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"output_summary={OUTPUT_SUMMARY.relative_to(REPO_ROOT)}")
    print("input_modified=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"District-name audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
