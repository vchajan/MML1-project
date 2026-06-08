from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.validation import make_valid


REPO_ROOT = Path(__file__).resolve().parents[1]
POINTS = REPO_ROOT / "data" / "reference" / "district_point_versions.csv"
POINTS_BY_YEAR = REPO_ROOT / "data" / "reference" / "district_points_by_crop_year.csv"
ASSIGNMENTS = REPO_ROOT / "data" / "reference" / "district_boundary_assignments_working.csv"
SHAPEFILES = {
    "Census 2001": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2001" / "2001_Dist.shp",
    "Census 2011": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2011" / "2011_Dist.shp",
}


def test_each_district_has_2001_and_2011_point_version() -> None:
    points = pd.read_csv(POINTS)
    counts = points.groupby("district_id")["requested_census_version"].nunique()
    assert len(counts) == 727
    assert counts.min() == 2
    assert counts.max() == 2


def test_each_point_lies_in_used_geometry() -> None:
    points = pd.read_csv(POINTS)
    layers = {version: gpd.read_file(path) for version, path in SHAPEFILES.items()}
    for row in points.itertuples(index=False):
        geom = layers[row.geometry_census_version].loc[int(row.boundary_feature_index)].geometry
        if not geom.is_valid:
            geom = make_valid(geom)
        assert geom.covers(wkt.loads(row.point_wkt))
        assert row.point_inside_geometry == 1


def test_invalid_geometry_repaired_only_in_derived_output() -> None:
    points = pd.read_csv(POINTS)
    assert points["geometry_repaired"].sum() == 1
    repaired = points[points["geometry_repaired"].eq(1)].iloc[0]
    assert repaired["geometry_repair_method"] == "shapely.make_valid"
    raw_2011 = gpd.read_file(SHAPEFILES["Census 2011"])
    assert (~raw_2011.geometry.is_valid).sum() == 1


def test_crop_year_2005_prefers_census_2001() -> None:
    points_by_year = pd.read_csv(POINTS_BY_YEAR)
    assert points_by_year[points_by_year["Crop_Year"].eq(2005)]["preferred_census_version"].eq("Census 2001").all()


def test_crop_year_2006_prefers_census_2011() -> None:
    points_by_year = pd.read_csv(POINTS_BY_YEAR)
    assert points_by_year[points_by_year["Crop_Year"].eq(2006)]["preferred_census_version"].eq("Census 2011").all()


def test_telangana_keeps_crop_state_and_historical_boundary_state() -> None:
    assignments = pd.read_csv(ASSIGNMENTS)
    telangana = assignments[assignments["canonical_state_name"].eq("Telangana")]
    assert not telangana.empty
    assert telangana["boundary_state_name"].eq("Andhra Pradesh").all()
    assert telangana["assignment_confidence"].eq("historical_fallback").all()


def test_punjab_s_uses_manual_override_in_census_2011() -> None:
    assignments = pd.read_csv(ASSIGNMENTS)
    row = assignments[
        assignments["raw_state_name"].eq("Punjab")
        & assignments["raw_district_name"].eq("S")
        & assignments["requested_census_version"].eq("Census 2011")
    ].iloc[0]
    assert row["canonical_district_name"] == "S.A.S NAGAR"
    assert row["boundary_district_name"] == "Sahibzada Ajit Singh Nagar"
    assert row["assignment_method"] == "confirmed_manual_override"
