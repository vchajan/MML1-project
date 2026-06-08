from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "datameet_districts"
LAYER_AUDIT_PATH = REPO_ROOT / "reports" / "district_boundary_layer_audit.csv"
INVENTORY_PATH = REPO_ROOT / "reports" / "boundary_district_name_inventory.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "district_boundary_source_summary.md"
REQUIRED_SHAPEFILE_EXTENSIONS = {".shp", ".shx", ".dbf"}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text.strip())
    return text.upper()


def compare_key(value: str) -> str:
    return re.sub(r"[\s\.\'`\-\(\)\[\]\{\}/&,]+", "", value)


def compact_column_name(column: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", column.upper())


def detect_columns(columns: list[str], kind: str) -> list[str]:
    compact = {column: compact_column_name(column) for column in columns}
    if kind == "district_name":
        preferred = {"DISTRICT", "DISTNAME", "DISTRICTNAME", "DTNAME", "DNAME", "DISTRICTN"}
        return [
            column
            for column, key in compact.items()
            if key in preferred
            or (("DIST" in key or key.startswith("DT")) and not any(token in key for token in ["CODE", "CD", "CEN"]))
        ]
    if kind == "state_name":
        preferred = {"STATE", "STATENAME", "STNAME", "STNM"}
        return [
            column
            for column, key in compact.items()
            if key in preferred
            or (("STATE" in key or key.startswith("ST")) and not any(token in key for token in ["CODE", "CD", "CEN"]))
        ]
    if kind == "district_code":
        return [
            column
            for column, key in compact.items()
            if ("DIST" in key or key.startswith("DT")) and any(token in key for token in ["CODE", "CD", "CEN"])
        ]
    if kind == "state_code":
        return [
            column
            for column, key in compact.items()
            if ("STATE" in key or key.startswith("ST")) and any(token in key for token in ["CODE", "CD", "CEN"])
        ]
    raise ValueError(f"Unknown column kind: {kind}")


def choose_column(candidates: list[str]) -> str:
    return candidates[0] if candidates else ""


def complete_shapefiles(census_dir: Path) -> list[Path]:
    if not census_dir.exists():
        raise RuntimeError(f"Missing census directory: {census_dir}")
    shapefiles = sorted(census_dir.rglob("*.shp"))
    if not shapefiles:
        raise RuntimeError(f"No shapefiles found under {census_dir}")
    for shapefile in shapefiles:
        base = shapefile.with_suffix("")
        missing = [ext for ext in REQUIRED_SHAPEFILE_EXTENSIONS if not base.with_suffix(ext).exists()]
        if missing:
            raise RuntimeError(f"Incomplete shapefile {shapefile}: missing {', '.join(missing)}")
    return shapefiles


def census_version_for_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "census_2001" in parts:
        return "Census 2001"
    if "census_2011" in parts:
        return "Census 2011"
    raise RuntimeError(f"Could not determine census version for {path}")


def count_duplicate_geometries(gdf: gpd.GeoDataFrame) -> int:
    geometry_keys = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            geometry_keys.append("")
        else:
            geometry_keys.append(geom.wkb_hex)
    series = pd.Series(geometry_keys)
    return int(series[series.ne("")].duplicated(keep=False).sum())


def audit_layer(path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError(f"GeoPandas read an empty layer: {path}")
    columns = [column for column in gdf.columns if column != gdf.geometry.name]
    district_name_columns = detect_columns(columns, "district_name")
    if not district_name_columns:
        raise RuntimeError(f"Could not identify a district-name column in {path}; columns={columns}")
    state_name_columns = detect_columns(columns, "state_name")
    district_code_columns = detect_columns(columns, "district_code")
    state_code_columns = detect_columns(columns, "state_code")

    district_column = choose_column(district_name_columns)
    state_column = choose_column(state_name_columns)
    district_code_column = choose_column(district_code_columns)
    state_code_column = choose_column(state_code_columns)
    bounds = gdf.total_bounds
    geometry_type_counts = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
    null_geometry_count = int(gdf.geometry.isna().sum())
    empty_geometry_count = int(gdf.geometry.apply(lambda geom: bool(geom is not None and geom.is_empty)).sum())
    invalid_geometry_count = int((~gdf.geometry.is_valid.fillna(False)).sum())
    attribute_columns = [column for column in gdf.columns if column != gdf.geometry.name]
    duplicate_attribute_row_count = int(gdf[attribute_columns].duplicated(keep=False).sum())
    census_version = census_version_for_path(path)

    audit = {
        "census_version": census_version,
        "file_path": path.relative_to(REPO_ROOT).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "feature_count": len(gdf),
        "column_names": ";".join(columns),
        "geometry_type_counts": ";".join(f"{key}:{value}" for key, value in geometry_type_counts.items()),
        "crs": str(gdf.crs) if gdf.crs else "",
        "crs_is_geographic": bool(gdf.crs and gdf.crs.is_geographic),
        "bounds_minx": bounds[0],
        "bounds_miny": bounds[1],
        "bounds_maxx": bounds[2],
        "bounds_maxy": bounds[3],
        "null_geometry_count": null_geometry_count,
        "empty_geometry_count": empty_geometry_count,
        "invalid_geometry_count": invalid_geometry_count,
        "duplicate_geometry_count": count_duplicate_geometries(gdf),
        "duplicate_attribute_row_count": duplicate_attribute_row_count,
        "state_name_columns_detected": ";".join(state_name_columns),
        "district_name_columns_detected": ";".join(district_name_columns),
        "state_code_columns_detected": ";".join(state_code_columns),
        "district_code_columns_detected": ";".join(district_code_columns),
    }

    raw_state = gdf[state_column] if state_column else pd.Series([""] * len(gdf))
    raw_district = gdf[district_column]
    raw_state_code = gdf[state_code_column] if state_code_column else pd.Series([""] * len(gdf))
    raw_district_code = gdf[district_code_column] if district_code_column else pd.Series([""] * len(gdf))
    normalized_state = raw_state.map(normalize_text)
    normalized_district = raw_district.map(normalize_text)

    inventory = pd.DataFrame(
        {
            "census_version": census_version,
            "source_layer": path.relative_to(REPO_ROOT).as_posix(),
            "source_feature_index": range(len(gdf)),
            "raw_state_name": raw_state,
            "raw_district_name": raw_district,
            "raw_state_code": raw_state_code,
            "raw_district_code": raw_district_code,
            "normalized_state_name": normalized_state,
            "normalized_district_name": normalized_district,
            "state_compare_key": normalized_state.map(compare_key),
            "district_compare_key": normalized_district.map(compare_key),
            "geometry_type": gdf.geometry.geom_type,
            "geometry_valid": gdf.geometry.is_valid.fillna(False),
            "geometry_area_native": gdf.geometry.area,
        }
    )
    return audit, inventory


def assert_external_files_not_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "data/external/datameet_districts/"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"External boundary files are tracked by Git:\n{result.stdout}")


def make_summary(layer_audit: pd.DataFrame, inventory: pd.DataFrame) -> str:
    lines = [
        "# District Boundary Source Summary",
        "",
        "DataMeet Census 2001 and Census 2011 district boundary layers were downloaded and technically audited.",
        "No district matching, fuzzy matching, representative-point generation, or coordinate assignment was performed.",
        "",
    ]
    for census_version in ["Census 2001", "Census 2011"]:
        layer_subset = layer_audit[layer_audit["census_version"].eq(census_version)]
        inventory_subset = inventory[inventory["census_version"].eq(census_version)]
        if layer_subset.empty:
            raise RuntimeError(f"Missing audit rows for {census_version}")
        issue_lines: list[str] = []
        invalid_count = int(layer_subset["invalid_geometry_count"].sum())
        empty_or_null = int(layer_subset["empty_geometry_count"].sum() + layer_subset["null_geometry_count"].sum())
        duplicate_geometry_count = int(layer_subset["duplicate_geometry_count"].sum())
        duplicate_attribute_count = int(layer_subset["duplicate_attribute_row_count"].sum())
        if invalid_count:
            issue_lines.append(f"{invalid_count} invalid geometries")
        if empty_or_null:
            issue_lines.append(f"{empty_or_null} empty or null geometries")
        if duplicate_geometry_count:
            issue_lines.append(f"{duplicate_geometry_count} duplicate geometry rows")
        if duplicate_attribute_count:
            issue_lines.append(f"{duplicate_attribute_count} duplicate attribute rows")
        if not issue_lines:
            issue_lines.append("No blocking technical issues detected")

        lines.extend(
            [
                f"## {census_version}",
                "",
                f"- Layers: {len(layer_subset)}",
                f"- Features: {int(layer_subset['feature_count'].sum())}",
                f"- Unique states: {inventory_subset['normalized_state_name'].replace('', pd.NA).dropna().nunique()}",
                f"- Unique district names: {inventory_subset['normalized_district_name'].replace('', pd.NA).dropna().nunique()}",
                f"- CRS: {'; '.join(sorted(set(layer_subset['crs'].astype(str))))}",
                f"- Invalid geometries: {invalid_count}",
                f"- Empty or null geometries: {empty_or_null}",
                f"- State name columns: {'; '.join(sorted(set(';'.join(layer_subset['state_name_columns_detected']).split(';')) - {''}))}",
                f"- District name columns: {'; '.join(sorted(set(';'.join(layer_subset['district_name_columns_detected']).split(';')) - {''}))}",
                f"- State code columns: {'; '.join(sorted(set(';'.join(layer_subset['state_code_columns_detected']).split(';')) - {''}))}",
                f"- District code columns: {'; '.join(sorted(set(';'.join(layer_subset['district_code_columns_detected']).split(';')) - {''}))}",
                (
                    "- Bounds: "
                    f"{layer_subset['bounds_minx'].min()}, {layer_subset['bounds_miny'].min()}, "
                    f"{layer_subset['bounds_maxx'].max()}, {layer_subset['bounds_maxy'].max()}"
                ),
                f"- Technical issues: {'; '.join(issue_lines)}",
                "- Matching status: not performed.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    shapefiles = []
    shapefiles.extend(complete_shapefiles(EXTERNAL_ROOT / "census_2001"))
    shapefiles.extend(complete_shapefiles(EXTERNAL_ROOT / "census_2011"))

    audit_rows: list[dict[str, object]] = []
    inventory_frames: list[pd.DataFrame] = []
    for shapefile in shapefiles:
        audit, inventory = audit_layer(shapefile)
        audit_rows.append(audit)
        inventory_frames.append(inventory)

    layer_audit = pd.DataFrame(audit_rows)
    inventory = pd.concat(inventory_frames, ignore_index=True)
    if layer_audit.empty or inventory.empty:
        raise RuntimeError("Boundary audit outputs are unexpectedly empty")

    assert_external_files_not_tracked()
    LAYER_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    layer_audit.to_csv(LAYER_AUDIT_PATH, index=False, lineterminator="\n")
    inventory.to_csv(INVENTORY_PATH, index=False, lineterminator="\n")
    SUMMARY_PATH.write_text(make_summary(layer_audit, inventory), encoding="utf-8", newline="\n")

    for census_version in ["Census 2001", "Census 2011"]:
        subset = layer_audit[layer_audit["census_version"].eq(census_version)]
        print(f"{census_version}: layers={len(subset)}, features={int(subset['feature_count'].sum())}")
    print(f"layer_audit={LAYER_AUDIT_PATH.relative_to(REPO_ROOT)}")
    print(f"inventory={INVENTORY_PATH.relative_to(REPO_ROOT)}")
    print("matching_performed=false")
    print("external_files_tracked=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
