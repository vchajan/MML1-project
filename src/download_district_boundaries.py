from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY_URL = "https://github.com/datameet/maps"
GITHUB_API_REPO = "https://api.github.com/repos/datameet/maps"
SOURCE_DIRS = {
    "census_2001": "Districts/Census_2001",
    "census_2011": "Districts/Census_2011",
}
EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "datameet_districts"
MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "boundary_sources" / "datameet_district_boundaries.json"
REQUIRED_SHAPEFILE_EXTENSIONS = {".shp", ".shx", ".dbf"}
OPTIONAL_SHAPEFILE_EXTENSIONS = {".prj", ".cpg"}


def http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "MML1-project-boundary-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MML1-project-boundary-downloader"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def get_source_commit_sha() -> str:
    repo_info = http_json(GITHUB_API_REPO)
    default_branch = repo_info["default_branch"]
    commit_info = http_json(f"{GITHUB_API_REPO}/commits/{default_branch}")
    return commit_info["sha"]


def manifest_files_are_valid(source_commit_sha: str) -> bool:
    if not MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("source_commit_sha") != source_commit_sha:
        return False
    for record in manifest.get("file_manifest", []):
        local_path = REPO_ROOT / record["local_path"]
        try:
            if not local_path.exists():
                return False
            if local_path.stat().st_size != int(record["size_bytes"]):
                return False
            if sha256_file(local_path) != record["sha256"]:
                return False
        except OSError:
            return False
    return True


def archive_member_to_source_path(member_name: str, source_commit_sha: str) -> str | None:
    prefix = f"maps-{source_commit_sha}/"
    if not member_name.startswith(prefix):
        return None
    return member_name[len(prefix) :]


def extract_source_dirs(archive_path: Path, staging_root: Path, source_commit_sha: str) -> None:
    extracted_any = {local_name: False for local_name in SOURCE_DIRS}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            source_path = archive_member_to_source_path(member.filename, source_commit_sha)
            if source_path is None:
                continue
            for local_name, source_dir in SOURCE_DIRS.items():
                source_prefix = f"{source_dir}/"
                if source_path.startswith(source_prefix):
                    relative_path = Path(source_path[len(source_prefix) :])
                    target = staging_root / local_name / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    extracted_any[local_name] = True
    missing = [name for name, found in extracted_any.items() if not found]
    if missing:
        raise RuntimeError(f"Archive did not contain expected source directories: {', '.join(missing)}")


def verify_shapefiles(root: Path) -> list[Path]:
    shapefiles = sorted(root.rglob("*.shp"))
    if not shapefiles:
        raise RuntimeError(f"No .shp files found under {root}")
    for shapefile in shapefiles:
        base = shapefile.with_suffix("")
        missing = [ext for ext in REQUIRED_SHAPEFILE_EXTENSIONS if not base.with_suffix(ext).exists()]
        if missing:
            raise RuntimeError(f"Incomplete shapefile {shapefile}: missing {', '.join(missing)}")
    return shapefiles


def build_file_manifest() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(EXTERNAL_ROOT.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "local_path": path.relative_to(REPO_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def write_source_manifest(source_commit_sha: str) -> None:
    file_manifest = build_file_manifest()
    manifest = {
        "source_name": "DataMeet maps historical district boundaries",
        "repository_url": SOURCE_REPOSITORY_URL,
        "source_commit_sha": source_commit_sha,
        "download_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_paths": SOURCE_DIRS,
        "local_paths": {
            "census_2001": "data/external/datameet_districts/census_2001",
            "census_2011": "data/external/datameet_districts/census_2011",
        },
        "license_name": "DataMeet maps attribution license; see source Districts folder",
        "license_reference": f"{SOURCE_REPOSITORY_URL}/tree/{source_commit_sha}/Districts",
        "attribution_text": (
            "District names and extents are derived from Administrative Atlas of India, Census of India; "
            "geometries are assembled using taluk boundaries and Bhuvan/GeoCommons-derived data as described "
            "by the DataMeet maps project."
        ),
        "geometry_caveat": (
            "These layers are a working historical reference assembled from multiple sources and are not "
            "legally authoritative administrative boundaries."
        ),
        "intended_project_use": (
            "Historical district-name matching and generation of representative points for weather queries."
        ),
        "required_shapefile_extensions": sorted(REQUIRED_SHAPEFILE_EXTENSIONS),
        "optional_shapefile_extensions": sorted(OPTIONAL_SHAPEFILE_EXTENSIONS),
        "file_manifest": file_manifest,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def replace_external_root(staging_root: Path) -> None:
    EXTERNAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if EXTERNAL_ROOT.exists():
        shutil.rmtree(EXTERNAL_ROOT)
    shutil.copytree(staging_root, EXTERNAL_ROOT)


def main() -> int:
    source_commit_sha = get_source_commit_sha()
    if manifest_files_are_valid(source_commit_sha):
        verify_shapefiles(EXTERNAL_ROOT / "census_2001")
        verify_shapefiles(EXTERNAL_ROOT / "census_2011")
        print("DataMeet district boundaries already present and verified")
        print(f"source_commit_sha={source_commit_sha}")
        print("download_skipped=true")
        return 0

    archive_url = f"{SOURCE_REPOSITORY_URL}/archive/{source_commit_sha}.zip"
    EXTERNAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="datameet_maps_", dir=EXTERNAL_ROOT.parent) as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "maps.zip"
        staging_root = temp_path / "datameet_districts"
        download_url(archive_url, archive_path)
        extract_source_dirs(archive_path, staging_root, source_commit_sha)
        verify_shapefiles(staging_root / "census_2001")
        verify_shapefiles(staging_root / "census_2011")
        replace_external_root(staging_root)

    verify_shapefiles(EXTERNAL_ROOT / "census_2001")
    verify_shapefiles(EXTERNAL_ROOT / "census_2011")
    write_source_manifest(source_commit_sha)
    print("DataMeet district boundaries downloaded")
    print(f"source_commit_sha={source_commit_sha}")
    print(f"local_root={EXTERNAL_ROOT.relative_to(REPO_ROOT)}")
    print(f"file_count={len(build_file_manifest())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
