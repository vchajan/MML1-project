from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.validation import make_valid


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS_PATH = REPO_ROOT / "data" / "reference" / "district_boundary_assignments_working.csv"
POINT_VERSIONS_PATH = REPO_ROOT / "data" / "reference" / "district_point_versions.csv"
POINTS_BY_YEAR_PATH = REPO_ROOT / "data" / "reference" / "district_points_by_crop_year.csv"
WEATHER_POINTS_PATH = REPO_ROOT / "data" / "reference" / "weather_points_unique.csv"
SHAPEFILES = {
    "Census 2001": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2001" / "2001_Dist.shp",
    "Census 2011": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2011" / "2011_Dist.shp",
}
YEARS = list(range(1997, 2015))


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:length].upper()}"


def polygonal_make_valid(geom):
    repaired = make_valid(geom)
    if isinstance(repaired, GeometryCollection):
        polygons = [part for part in repaired.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        if not polygons:
            return repaired
        return MultiPolygon([poly for geom_part in polygons for poly in (geom_part.geoms if geom_part.geom_type == "MultiPolygon" else [geom_part])])
    return repaired


def load_layers() -> dict[str, gpd.GeoDataFrame]:
    layers = {}
    for census_version, path in SHAPEFILES.items():
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"Empty shapefile: {path}")
        if gdf.crs is None or str(gdf.crs) != "EPSG:4326":
            raise ValueError(f"{census_version} must be EPSG:4326, found {gdf.crs}")
        layers[census_version] = gdf
    return layers


def build_point_versions(assignments: pd.DataFrame, layers: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in assignments.iterrows():
        gdf = layers[row["geometry_census_version"]]
        feature_index = int(row["boundary_feature_index"])
        if feature_index not in gdf.index:
            raise ValueError(f"Missing boundary feature index {feature_index} in {row['geometry_census_version']}")
        geom = gdf.loc[feature_index].geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"Null or empty geometry for {row['district_id']} {row['requested_census_version']}")
        geometry_was_valid = bool(geom.is_valid)
        geometry_repaired = False
        repair_method = ""
        if not geometry_was_valid:
            geom = polygonal_make_valid(geom)
            geometry_repaired = True
            repair_method = "shapely.make_valid"
        if geom is None or geom.is_empty:
            raise ValueError(f"Geometry repair produced empty geometry for {row['district_id']}")
        point = geom.representative_point()
        point_inside = bool(geom.covers(point))
        if not point_inside:
            raise ValueError(f"Representative point is outside geometry for {row['district_id']} {row['requested_census_version']}")
        point_id = stable_id("DPV", row["district_id"], row["requested_census_version"], row["geometry_census_version"], feature_index)
        rows.append(
            {
                "district_point_version_id": point_id,
                "district_id": row["district_id"],
                "requested_census_version": row["requested_census_version"],
                "geometry_census_version": row["geometry_census_version"],
                "boundary_feature_index": feature_index,
                "raw_state_name": row["raw_state_name"],
                "raw_district_name": row["raw_district_name"],
                "canonical_state_name": row["canonical_state_name"],
                "canonical_district_name": row["canonical_district_name"],
                "boundary_state_name": row["boundary_state_name"],
                "boundary_district_name": row["boundary_district_name"],
                "latitude": round(point.y, 8),
                "longitude": round(point.x, 8),
                "point_wkt": point.wkt,
                "point_method": "geometry.representative_point",
                "assignment_method": row["assignment_method"],
                "assignment_confidence": row["assignment_confidence"],
                "shared_parent_boundary": row["shared_parent_boundary"],
                "geometry_was_valid": int(geometry_was_valid),
                "geometry_repaired": int(geometry_repaired),
                "geometry_repair_method": repair_method,
                "point_inside_geometry": int(point_inside),
            }
        )
    points = pd.DataFrame(rows)
    if len(points) != 1454:
        raise ValueError(f"Expected 1454 point version rows, found {len(points)}")
    if points["point_inside_geometry"].ne(1).any():
        raise ValueError("Some points are outside their geometry")
    return points


def build_points_by_year(points: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assignment_flags = assignments[
        ["district_id", "requested_census_version", "alternate_layer_used", "shared_parent_boundary"]
    ].copy()
    points = points.merge(assignment_flags, on=["district_id", "requested_census_version", "shared_parent_boundary"], how="left")
    for _, point in points.iterrows():
        for year in YEARS:
            preferred = "Census 2001" if year <= 2005 else "Census 2011"
            if point["requested_census_version"] != preferred:
                continue
            weather_point_id = stable_id("WPT", f"{float(point['latitude']):.6f}", f"{float(point['longitude']):.6f}", length=20)
            rows.append(
                {
                    "district_year_point_id": stable_id("DYP", point["district_id"], year),
                    "district_id": point["district_id"],
                    "Crop_Year": year,
                    "preferred_census_version": preferred,
                    "used_census_version": point["geometry_census_version"],
                    "district_point_version_id": point["district_point_version_id"],
                    "weather_point_id": weather_point_id,
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                    "assignment_method": point["assignment_method"],
                    "assignment_confidence": point["assignment_confidence"],
                    "alternate_layer_used": int(point["alternate_layer_used"]),
                    "shared_parent_boundary": int(point["shared_parent_boundary"]),
                }
            )
    by_year = pd.DataFrame(rows)
    if len(by_year) != 727 * 18:
        raise ValueError(f"Expected 13086 district-year points, found {len(by_year)}")
    return by_year


def build_weather_points(points_by_year: pd.DataFrame) -> pd.DataFrame:
    grouped = points_by_year.groupby("weather_point_id").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        district_point_versions_count=("district_point_version_id", "nunique"),
        district_ids_count=("district_id", "nunique"),
        first_crop_year=("Crop_Year", "min"),
        last_crop_year=("Crop_Year", "max"),
    )
    return grouped.reset_index()


def main() -> int:
    assignments = pd.read_csv(ASSIGNMENTS_PATH, keep_default_na=False)
    layers = load_layers()
    point_versions = build_point_versions(assignments, layers)
    points_by_year = build_points_by_year(point_versions, assignments)
    weather_points = build_weather_points(points_by_year)
    POINT_VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    point_versions.to_csv(POINT_VERSIONS_PATH, index=False, lineterminator="\n")
    points_by_year.to_csv(POINTS_BY_YEAR_PATH, index=False, lineterminator="\n")
    weather_points.to_csv(WEATHER_POINTS_PATH, index=False, lineterminator="\n")
    print(f"district_point_versions={len(point_versions)}")
    print(f"district_year_points={len(points_by_year)}")
    print(f"unique_weather_points={len(weather_points)}")
    print(f"geometry_repaired={int(point_versions['geometry_repaired'].sum())}")
    print(f"points_outside={int(point_versions['point_inside_geometry'].ne(1).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
