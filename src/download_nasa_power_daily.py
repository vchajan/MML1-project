from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = REPO_ROOT / "data" / "interim" / "weather_windows_1997_2014.parquet"
POINTS_PATH = REPO_ROOT / "data" / "reference" / "weather_points_unique.csv"
CACHE_DIR = REPO_ROOT / "data" / "interim" / "weather_daily"
STATUS_PATH = REPO_ROOT / "reports" / "nasa_power_download_status.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "nasa_power_download_summary.md"
ERRORS_PATH = REPO_ROOT / "reports" / "nasa_power_download_errors.csv"
MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "nasa_power_request_manifest.json"

ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]
COMMUNITY = "AG"
TIME_STANDARD = "LST"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
SENTINELS = {-999, -999.0, -9999, -9999.0}


def yyyymmdd(value: str | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def expected_date_frame(start_date: str, end_date: str) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(start_date, end_date).strftime("%Y-%m-%d")})


def cache_path(weather_point_id: str) -> Path:
    return CACHE_DIR / f"{weather_point_id}.csv"


def normalize_value(value: object) -> float | None:
    if value in SENTINELS:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric in SENTINELS or math.isclose(numeric, -999.0) or math.isclose(numeric, -9999.0):
        return None
    return numeric


def parse_nasa_response(payload: dict, weather_point_id: str, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    parameters = payload.get("properties", {}).get("parameter", {})
    missing = [param for param in PARAMETERS if param not in parameters]
    if missing:
        raise ValueError(f"NASA response missing parameters: {', '.join(missing)}")
    frame = expected_date_frame(start_date, end_date)
    frame["weather_point_id"] = weather_point_id
    frame["latitude"] = latitude
    frame["longitude"] = longitude
    frame["date_key"] = frame["date"].str.replace("-", "", regex=False)
    for param in PARAMETERS:
        values = parameters[param]
        frame[param] = frame["date_key"].map(lambda key, mapping=values: normalize_value(mapping.get(key)))
    frame = frame.drop(columns=["date_key"])
    return frame[["weather_point_id", "date", "latitude", "longitude", *PARAMETERS]]


def cache_is_complete(path: Path, weather_point_id: str, start_date: str, end_date: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        cached = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive path
        return False, f"read_error: {exc}"
    required = {"weather_point_id", "date", "latitude", "longitude", *PARAMETERS}
    missing = sorted(required - set(cached.columns))
    if missing:
        return False, f"missing_columns: {', '.join(missing)}"
    expected_dates = expected_date_frame(start_date, end_date)["date"]
    if len(cached) != len(expected_dates):
        return False, f"row_count {len(cached)} != {len(expected_dates)}"
    if str(cached["date"].min()) != str(expected_dates.min()) or str(cached["date"].max()) != str(expected_dates.max()):
        return False, "date_range_mismatch"
    if cached["weather_point_id"].astype(str).nunique() != 1 or str(cached["weather_point_id"].iloc[0]) != weather_point_id:
        return False, "weather_point_id_mismatch"
    return True, "complete"


def request_params(latitude: float, longitude: float, start_date: str, end_date: str) -> dict[str, str | float]:
    return {
        "parameters": ",".join(PARAMETERS),
        "community": COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": yyyymmdd(start_date),
        "end": yyyymmdd(end_date),
        "format": "JSON",
        "time-standard": TIME_STANDARD,
    }


def fetch_with_retry(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    sleep_func: Callable[[float], None] = time.sleep,
    max_retries: int = MAX_RETRIES,
) -> tuple[dict, int]:
    last_error = ""
    last_status = 0
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                ENDPOINT,
                params=request_params(latitude, longitude, start_date, end_date),
                timeout=TIMEOUT_SECONDS,
            )
            last_status = response.status_code
            if response.status_code == 200:
                return response.json(), response.status_code
            last_error = f"HTTP {response.status_code}: {response.text[:250]}"
        except Exception as exc:  # pragma: no cover - exercised through tests with fake session
            last_error = str(exc)
        if attempt < max_retries:
            sleep_func(max(1.0, 2 ** (attempt - 1)))
    raise RuntimeError(last_error or f"NASA request failed with status {last_status}")


def load_plan() -> tuple[pd.DataFrame, str, str]:
    windows = pd.read_parquet(WINDOWS_PATH)
    points = pd.read_csv(POINTS_PATH)
    start_date = str(windows["start_date"].min())
    end_date = str(windows["end_date"].max())
    if len(points) == 0:
        raise ValueError("No weather points to download")
    return points.sort_values("weather_point_id"), start_date, end_date


def write_manifest(point_count: int, start_date: str, end_date: str) -> None:
    manifest = {
        "endpoint": ENDPOINT,
        "community": COMMUNITY,
        "time_standard": TIME_STANDARD,
        "parameters": PARAMETERS,
        "date_range": {"start_date": start_date, "end_date": end_date},
        "download_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "point_count": point_count,
        "cache_directory": "data/interim/weather_daily",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_reports(status_rows: list[dict[str, object]], error_rows: list[dict[str, object]], start_date: str, end_date: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame(status_rows)
    errors = pd.DataFrame(error_rows, columns=["weather_point_id", "latitude", "longitude", "http_status", "error"])
    status.to_csv(STATUS_PATH, index=False, lineterminator="\n")
    errors.to_csv(ERRORS_PATH, index=False, lineterminator="\n")
    counts = status["status"].value_counts()
    lines = [
        "# NASA POWER Download Summary",
        "",
        f"- Points requested: {len(status)}",
        f"- Downloaded: {int(counts.get('downloaded', 0))}",
        f"- Cache reused: {int(counts.get('cache_reused', 0))}",
        f"- Failed: {int(counts.get('failed', 0))}",
        f"- Date range: {start_date} to {end_date}",
        f"- Parameters: {', '.join(PARAMETERS)}",
        f"- Endpoint: {ENDPOINT}",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def download_all(force_redownload: bool = False) -> int:
    points, start_date, end_date = load_plan()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    status_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    last_request_at = 0.0
    for point in points.itertuples(index=False):
        point_id = str(point.weather_point_id)
        path = cache_path(point_id)
        complete, cache_message = cache_is_complete(path, point_id, start_date, end_date)
        if complete and not force_redownload:
            status_rows.append(
                {
                    "weather_point_id": point_id,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "status": "cache_reused",
                    "http_status": "",
                    "rows": pd.read_csv(path, usecols=["date"]).shape[0],
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": cache_message,
                }
            )
            continue
        if path.exists():
            path.unlink()
        elapsed = time.time() - last_request_at
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
        try:
            payload, http_status = fetch_with_retry(session, point.latitude, point.longitude, start_date, end_date)
            last_request_at = time.time()
            frame = parse_nasa_response(payload, point_id, point.latitude, point.longitude, start_date, end_date)
            frame.to_csv(path, index=False, lineterminator="\n")
            complete, message = cache_is_complete(path, point_id, start_date, end_date)
            if not complete:
                raise RuntimeError(f"Downloaded cache is incomplete: {message}")
            status_rows.append(
                {
                    "weather_point_id": point_id,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "status": "downloaded",
                    "http_status": http_status,
                    "rows": len(frame),
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": "ok",
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "weather_point_id": point_id,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "status": "failed",
                    "http_status": "",
                    "rows": 0,
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": str(exc),
                }
            )
            error_rows.append(
                {
                    "weather_point_id": point_id,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "http_status": "",
                    "error": str(exc),
                }
            )
    write_manifest(len(points), start_date, end_date)
    write_reports(status_rows, error_rows, start_date, end_date)
    failed = [row for row in status_rows if row["status"] == "failed"]
    print(f"points_requested={len(status_rows)}")
    print(f"downloaded={sum(row['status'] == 'downloaded' for row in status_rows)}")
    print(f"cache_reused={sum(row['status'] == 'cache_reused' for row in status_rows)}")
    print(f"failed={len(failed)}")
    print(f"date_range={start_date}..{end_date}")
    if failed:
        raise SystemExit("NASA POWER download failed for at least one point; aggregation not allowed")
    return 0


def main() -> int:
    return download_all(force_redownload=False)


if __name__ == "__main__":
    raise SystemExit(main())
