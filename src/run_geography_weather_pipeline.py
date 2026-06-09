from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from download_nasa_power_daily import CACHE_DIR, MANIFEST_PATH, cache_is_complete


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SCRIPT = REPO_ROOT / "src" / "aggregate_crop_weather_windows.py"
JOIN_SCRIPT = REPO_ROOT / "src" / "build_crop_weather_dataset.py"
WEATHER_POINTS_PATH = REPO_ROOT / "data" / "reference" / "weather_points_unique.csv"
FEATURES_PATH = REPO_ROOT / "data" / "interim" / "weather_features_by_window_1997_2014.parquet"
CROP_WEATHER_PATH = REPO_ROOT / "data" / "interim" / "crop_weather_dataset_1997_2014.parquet"
EXPECTED_CACHE_FILES = 701


def verify_weather_cache() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing NASA POWER manifest: {MANIFEST_PATH}")
    if not WEATHER_POINTS_PATH.exists():
        raise FileNotFoundError(f"Missing weather points table: {WEATHER_POINTS_PATH}")
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing NASA POWER cache directory: {CACHE_DIR}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    start_date = manifest["date_range"]["start_date"]
    end_date = manifest["date_range"]["end_date"]
    points = pd.read_csv(WEATHER_POINTS_PATH)
    csv_files = list(CACHE_DIR.glob("*.csv"))
    if len(points) != EXPECTED_CACHE_FILES:
        raise ValueError(f"Expected {EXPECTED_CACHE_FILES} weather points, found {len(points)}")
    if len(csv_files) != EXPECTED_CACHE_FILES:
        raise ValueError(f"Expected {EXPECTED_CACHE_FILES} cache CSV files, found {len(csv_files)}")

    incomplete: list[str] = []
    for point_id in points["weather_point_id"].astype(str):
        path = CACHE_DIR / f"{point_id}.csv"
        complete, message = cache_is_complete(path, point_id, start_date, end_date)
        if not complete:
            incomplete.append(f"{point_id}: {message}")
    if incomplete:
        preview = "; ".join(incomplete[:5])
        raise ValueError(f"Incomplete NASA POWER cache files: {preview}")
    print(f"cache_verified={len(csv_files)}")
    print(f"cache_date_range={start_date}..{end_date}")


def run_step(name: str, script: Path, output_paths: list[Path]) -> None:
    start = time.perf_counter()
    print(f"START {name}")
    result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"FAILED {name} elapsed_seconds={elapsed:.2f}")
        raise SystemExit(result.returncode)
    print(f"DONE {name} elapsed_seconds={elapsed:.2f}")
    for path in output_paths:
        print(f"output={path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local geography/weather pipeline steps without redownloading NASA data.")
    parser.add_argument("--skip-download", action="store_true", help="Verify existing NASA POWER cache and do not download data.")
    parser.add_argument("--only-aggregate", action="store_true", help="Run only weather window aggregation.")
    parser.add_argument("--only-join", action="store_true", help="Run only crop/weather join.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.only_aggregate and args.only_join:
        raise SystemExit("--only-aggregate and --only-join cannot be combined")

    verify_weather_cache()
    if args.skip_download:
        print("download=skipped")
    else:
        print("download=not_run_existing_cache_only")

    if args.only_join:
        run_step("crop_weather_join", JOIN_SCRIPT, [CROP_WEATHER_PATH])
        return 0
    if args.only_aggregate:
        run_step("weather_aggregation", AGGREGATE_SCRIPT, [FEATURES_PATH])
        return 0

    run_step("weather_aggregation", AGGREGATE_SCRIPT, [FEATURES_PATH])
    run_step("crop_weather_join", JOIN_SCRIPT, [CROP_WEATHER_PATH])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
