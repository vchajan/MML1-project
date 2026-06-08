from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
POINTS_PATH = REPO_ROOT / "data" / "reference" / "district_point_versions.csv"
MAP_PATH = REPO_ROOT / "maps" / "district_points_working.html"
SHAPEFILES = {
    "Census 2001": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2001" / "2001_Dist.shp",
    "Census 2011": REPO_ROOT / "data" / "external" / "datameet_districts" / "census_2011" / "2011_Dist.shp",
}
COLOR_BY_CONFIDENCE = {
    "confirmed": "#1b9e77",
    "working_strong": "#377eb8",
    "working_fallback": "#ff7f00",
    "historical_fallback": "#984ea3",
}


def add_polygon_layer(map_obj: folium.Map, census_version: str, path: Path, color: str) -> None:
    gdf = gpd.read_file(path)
    gdf = gdf[["ST_NM", "DISTRICT", "geometry"]].copy()
    folium.GeoJson(
        gdf.__geo_interface__,
        name=f"{census_version} polygons",
        style_function=lambda _feature, line_color=color: {
            "fillColor": "transparent",
            "color": line_color,
            "weight": 1,
            "fillOpacity": 0.0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["ST_NM", "DISTRICT"], aliases=["State", "District"]),
    ).add_to(map_obj)


def popup_html(row: pd.Series) -> str:
    return "<br>".join(
        [
            f"<b>district_id</b>: {row['district_id']}",
            f"<b>raw</b>: {row['raw_state_name']} / {row['raw_district_name']}",
            f"<b>canonical</b>: {row['canonical_state_name']} / {row['canonical_district_name']}",
            f"<b>boundary</b>: {row['boundary_state_name']} / {row['boundary_district_name']}",
            f"<b>requested census</b>: {row['requested_census_version']}",
            f"<b>geometry census</b>: {row['geometry_census_version']}",
            f"<b>assignment method</b>: {row['assignment_method']}",
            f"<b>assignment confidence</b>: {row['assignment_confidence']}",
            f"<b>shared parent boundary</b>: {row['shared_parent_boundary']}",
        ]
    )


def add_points(map_obj: folium.Map, points: pd.DataFrame) -> None:
    for confidence, color in COLOR_BY_CONFIDENCE.items():
        subset = points[points["assignment_confidence"].eq(confidence)]
        group = folium.FeatureGroup(name=f"points {confidence}", show=True)
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html(row), max_width=420),
            ).add_to(group)
        group.add_to(map_obj)


def main() -> int:
    points = pd.read_csv(POINTS_PATH)
    if len(points) != 1454:
        raise ValueError(f"Expected 1454 point versions, found {len(points)}")
    center = [float(points["latitude"].mean()), float(points["longitude"].mean())]
    map_obj = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    add_polygon_layer(map_obj, "Census 2001", SHAPEFILES["Census 2001"], "#4daf4a")
    add_polygon_layer(map_obj, "Census 2011", SHAPEFILES["Census 2011"], "#377eb8")
    add_points(map_obj, points)
    folium.LayerControl(collapsed=False).add_to(map_obj)
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(MAP_PATH)
    print(f"map={MAP_PATH.relative_to(REPO_ROOT)}")
    print(f"points={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
