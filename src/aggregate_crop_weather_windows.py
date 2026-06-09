from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = REPO_ROOT / "data" / "interim" / "weather_windows_1997_2014.parquet"
CACHE_DIR = REPO_ROOT / "data" / "interim" / "weather_daily"
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "weather_features_by_window_1997_2014.parquet"
STATUS_PATH = REPO_ROOT / "reports" / "weather_window_aggregation_status.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "weather_window_aggregation_summary.md"
ERRORS_PATH = REPO_ROOT / "reports" / "weather_window_aggregation_errors.csv"

NASA_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]
FEATURE_COLUMNS = [
    "weather_days_expected",
    "weather_days_present",
    "weather_coverage_ratio",
    "weather_window_valid",
    "rain_sum_mm",
    "rain_mean_mm",
    "rainy_days_ge1mm",
    "dry_days_lt1mm",
    "heavy_rain_days_ge20mm",
    "longest_dry_spell_days",
    "longest_wet_spell_days",
    "temp_mean_c",
    "temp_max_mean_c",
    "temp_min_mean_c",
    "temp_max_absolute_c",
    "temp_min_absolute_c",
    "hot_days_tmax_ge35c",
    "cold_days_tmin_lt10c",
    "humidity_mean_pct",
    "solar_radiation_mean",
    "wind_speed_mean",
    "first_25pct_days",
    "first_25pct_rain_sum_mm",
    "first_25pct_temp_mean_c",
    "first_25pct_longest_dry_spell_days",
    "last_25pct_days",
    "last_25pct_rain_sum_mm",
    "last_25pct_temp_mean_c",
    "last_25pct_heavy_rain_days",
]


def longest_true_run(mask: pd.Series | list[bool]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calculate_longest_dry_spell(precipitation: pd.Series) -> int:
    return longest_true_run(precipitation.lt(1.0) & precipitation.notna())


def calculate_longest_wet_spell(precipitation: pd.Series) -> int:
    return longest_true_run(precipitation.ge(1.0) & precipitation.notna())


def select_first_last_quarter(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe.empty:
        return dataframe.copy(), dataframe.copy()
    ordered = dataframe.sort_values("date")
    quarter_days = math.ceil(len(ordered) * 0.25)
    return ordered.head(quarter_days).copy(), ordered.tail(quarter_days).copy()


def _sum_or_nan(series: pd.Series) -> float:
    return float(series.sum(skipna=True, min_count=1))


def _int_count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def _longest_true_run_array(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def aggregate_weather_window(dataframe: pd.DataFrame, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> dict[str, object]:
    if "date" not in dataframe.columns:
        raise ValueError("Daily weather dataframe must contain a date column")
    daily = dataframe.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end < start:
        raise ValueError(f"Invalid weather window: {start_date} > {end_date}")
    window = daily.loc[(daily["date"] >= start) & (daily["date"] <= end)].sort_values("date").copy()
    expected_days = int((end - start).days + 1)
    present_days = int(len(window))
    coverage = present_days / expected_days if expected_days else 0.0
    first_quarter, last_quarter = select_first_last_quarter(window)

    precip = window["PRECTOTCORR"]
    tmean = window["T2M"]
    tmax = window["T2M_MAX"]
    tmin = window["T2M_MIN"]
    first_precip = first_quarter["PRECTOTCORR"]
    last_precip = last_quarter["PRECTOTCORR"]

    return {
        "weather_days_expected": expected_days,
        "weather_days_present": present_days,
        "weather_coverage_ratio": coverage,
        "weather_window_valid": bool(coverage >= 0.98),
        "rain_sum_mm": _sum_or_nan(precip),
        "rain_mean_mm": float(precip.mean()),
        "rainy_days_ge1mm": _int_count(precip.ge(1.0) & precip.notna()),
        "dry_days_lt1mm": _int_count(precip.lt(1.0) & precip.notna()),
        "heavy_rain_days_ge20mm": _int_count(precip.ge(20.0) & precip.notna()),
        "longest_dry_spell_days": calculate_longest_dry_spell(precip),
        "longest_wet_spell_days": calculate_longest_wet_spell(precip),
        "temp_mean_c": float(tmean.mean()),
        "temp_max_mean_c": float(tmax.mean()),
        "temp_min_mean_c": float(tmin.mean()),
        "temp_max_absolute_c": float(tmax.max()),
        "temp_min_absolute_c": float(tmin.min()),
        "hot_days_tmax_ge35c": _int_count(tmax.ge(35.0) & tmax.notna()),
        "cold_days_tmin_lt10c": _int_count(tmin.lt(10.0) & tmin.notna()),
        "humidity_mean_pct": float(window["RH2M"].mean()),
        "solar_radiation_mean": float(window["ALLSKY_SFC_SW_DWN"].mean()),
        "wind_speed_mean": float(window["WS2M"].mean()),
        "first_25pct_days": int(len(first_quarter)),
        "first_25pct_rain_sum_mm": _sum_or_nan(first_precip),
        "first_25pct_temp_mean_c": float(first_quarter["T2M"].mean()),
        "first_25pct_longest_dry_spell_days": calculate_longest_dry_spell(first_precip),
        "last_25pct_days": int(len(last_quarter)),
        "last_25pct_rain_sum_mm": _sum_or_nan(last_precip),
        "last_25pct_temp_mean_c": float(last_quarter["T2M"].mean()),
        "last_25pct_heavy_rain_days": _int_count(last_precip.ge(20.0) & last_precip.notna()),
    }


def read_daily_cache(weather_point_id: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    path = cache_dir / f"{weather_point_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing NASA POWER cache file: {path}")
    frame = pd.read_csv(path)
    required = {"weather_point_id", "date", *NASA_PARAMETERS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    for column in NASA_PARAMETERS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date")


def _cumulative(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = ~np.isnan(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid.astype(int))))
    missing = np.concatenate(([0], np.cumsum((~valid).astype(int))))
    return sums, counts, missing


def _range_count(cumulative: np.ndarray, start: int, end: int) -> int:
    return int(cumulative[end] - cumulative[start])


def _range_sum(cumulative_sum: np.ndarray, cumulative_count: np.ndarray, start: int, end: int) -> float:
    count = _range_count(cumulative_count, start, end)
    if count == 0:
        return float("nan")
    return float(cumulative_sum[end] - cumulative_sum[start])


def _range_mean(cumulative_sum: np.ndarray, cumulative_count: np.ndarray, start: int, end: int) -> float:
    count = _range_count(cumulative_count, start, end)
    if count == 0:
        return float("nan")
    return float((cumulative_sum[end] - cumulative_sum[start]) / count)


def _range_min(values: np.ndarray, cumulative_count: np.ndarray, start: int, end: int) -> float:
    if _range_count(cumulative_count, start, end) == 0:
        return float("nan")
    return float(np.nanmin(values[start:end]))


def _range_max(values: np.ndarray, cumulative_count: np.ndarray, start: int, end: int) -> float:
    if _range_count(cumulative_count, start, end) == 0:
        return float("nan")
    return float(np.nanmax(values[start:end]))


def _build_daily_arrays(daily: pd.DataFrame) -> dict[str, object]:
    dates = daily["date"].dt.floor("D").to_numpy(dtype="datetime64[D]")
    arrays: dict[str, object] = {"dates": dates}
    for param in NASA_PARAMETERS:
        values = daily[param].to_numpy(dtype=float)
        sums, counts, missing = _cumulative(values)
        arrays[param] = values
        arrays[f"{param}_sum"] = sums
        arrays[f"{param}_count"] = counts
        arrays[f"{param}_missing"] = missing

    precip = arrays["PRECTOTCORR"]
    tmax = arrays["T2M_MAX"]
    tmin = arrays["T2M_MIN"]
    assert isinstance(precip, np.ndarray)
    assert isinstance(tmax, np.ndarray)
    assert isinstance(tmin, np.ndarray)
    arrays["dry_cumsum"] = np.concatenate(([0], np.cumsum(((precip < 1.0) & ~np.isnan(precip)).astype(int))))
    arrays["wet_cumsum"] = np.concatenate(([0], np.cumsum(((precip >= 1.0) & ~np.isnan(precip)).astype(int))))
    arrays["heavy_cumsum"] = np.concatenate(([0], np.cumsum(((precip >= 20.0) & ~np.isnan(precip)).astype(int))))
    arrays["hot_cumsum"] = np.concatenate(([0], np.cumsum(((tmax >= 35.0) & ~np.isnan(tmax)).astype(int))))
    arrays["cold_cumsum"] = np.concatenate(([0], np.cumsum(((tmin < 10.0) & ~np.isnan(tmin)).astype(int))))
    return arrays


def _aggregate_from_arrays(arrays: dict[str, object], start_date: object, end_date: object) -> tuple[dict[str, object], dict[str, int]]:
    dates = arrays["dates"]
    assert isinstance(dates, np.ndarray)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end < start:
        raise ValueError(f"Invalid weather window: {start_date} > {end_date}")
    start_day = np.datetime64(start.date(), "D")
    end_day = np.datetime64(end.date(), "D")
    left = int(np.searchsorted(dates, start_day, side="left"))
    right = int(np.searchsorted(dates, end_day, side="right"))
    expected_days = int((end - start).days + 1)
    present_days = right - left
    coverage = present_days / expected_days if expected_days else 0.0

    precip = arrays["PRECTOTCORR"]
    tmax = arrays["T2M_MAX"]
    tmin = arrays["T2M_MIN"]
    assert isinstance(precip, np.ndarray)
    assert isinstance(tmax, np.ndarray)
    assert isinstance(tmin, np.ndarray)
    quarter_days = math.ceil(present_days * 0.25) if present_days else 0
    first_left, first_right = left, left + quarter_days
    last_left, last_right = right - quarter_days, right
    dry_mask = (precip < 1.0) & ~np.isnan(precip)
    wet_mask = (precip >= 1.0) & ~np.isnan(precip)

    features = {
        "weather_days_expected": expected_days,
        "weather_days_present": present_days,
        "weather_coverage_ratio": coverage,
        "weather_window_valid": bool(coverage >= 0.98),
        "rain_sum_mm": _range_sum(arrays["PRECTOTCORR_sum"], arrays["PRECTOTCORR_count"], left, right),
        "rain_mean_mm": _range_mean(arrays["PRECTOTCORR_sum"], arrays["PRECTOTCORR_count"], left, right),
        "rainy_days_ge1mm": _range_count(arrays["wet_cumsum"], left, right),
        "dry_days_lt1mm": _range_count(arrays["dry_cumsum"], left, right),
        "heavy_rain_days_ge20mm": _range_count(arrays["heavy_cumsum"], left, right),
        "longest_dry_spell_days": _longest_true_run_array(dry_mask[left:right]),
        "longest_wet_spell_days": _longest_true_run_array(wet_mask[left:right]),
        "temp_mean_c": _range_mean(arrays["T2M_sum"], arrays["T2M_count"], left, right),
        "temp_max_mean_c": _range_mean(arrays["T2M_MAX_sum"], arrays["T2M_MAX_count"], left, right),
        "temp_min_mean_c": _range_mean(arrays["T2M_MIN_sum"], arrays["T2M_MIN_count"], left, right),
        "temp_max_absolute_c": _range_max(tmax, arrays["T2M_MAX_count"], left, right),
        "temp_min_absolute_c": _range_min(tmin, arrays["T2M_MIN_count"], left, right),
        "hot_days_tmax_ge35c": _range_count(arrays["hot_cumsum"], left, right),
        "cold_days_tmin_lt10c": _range_count(arrays["cold_cumsum"], left, right),
        "humidity_mean_pct": _range_mean(arrays["RH2M_sum"], arrays["RH2M_count"], left, right),
        "solar_radiation_mean": _range_mean(arrays["ALLSKY_SFC_SW_DWN_sum"], arrays["ALLSKY_SFC_SW_DWN_count"], left, right),
        "wind_speed_mean": _range_mean(arrays["WS2M_sum"], arrays["WS2M_count"], left, right),
        "first_25pct_days": quarter_days,
        "first_25pct_rain_sum_mm": _range_sum(arrays["PRECTOTCORR_sum"], arrays["PRECTOTCORR_count"], first_left, first_right),
        "first_25pct_temp_mean_c": _range_mean(arrays["T2M_sum"], arrays["T2M_count"], first_left, first_right),
        "first_25pct_longest_dry_spell_days": _longest_true_run_array(dry_mask[first_left:first_right]),
        "last_25pct_days": quarter_days,
        "last_25pct_rain_sum_mm": _range_sum(arrays["PRECTOTCORR_sum"], arrays["PRECTOTCORR_count"], last_left, last_right),
        "last_25pct_temp_mean_c": _range_mean(arrays["T2M_sum"], arrays["T2M_count"], last_left, last_right),
        "last_25pct_heavy_rain_days": _range_count(arrays["heavy_cumsum"], last_left, last_right),
    }
    missing_counts = {
        f"missing_{param}": _range_count(arrays[f"{param}_missing"], left, right)
        for param in NASA_PARAMETERS
    }
    return features, missing_counts


def aggregate_windows(windows: pd.DataFrame, cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"weather_window_id", "weather_point_id", "start_date", "end_date", "expected_days"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Weather windows are missing columns: {', '.join(missing)}")
    if not windows["weather_window_id"].is_unique:
        raise ValueError("weather_window_id must be unique in weather windows")

    feature_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    ordered_windows = windows.sort_values(["weather_point_id", "start_date", "end_date", "weather_window_id"])
    for point_id, point_windows in ordered_windows.groupby("weather_point_id", sort=False):
        try:
            daily = read_daily_cache(str(point_id), cache_dir)
            arrays = _build_daily_arrays(daily)
        except Exception as exc:
            for window in point_windows.itertuples(index=False):
                row = window._asdict()
                window_id = str(row["weather_window_id"])
                error_rows.append(
                    {
                        "weather_window_id": window_id,
                        "weather_point_id": str(point_id),
                        "start_date": str(row["start_date"]),
                        "end_date": str(row["end_date"]),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                status_rows.append(
                    {
                        "weather_window_id": window_id,
                        "weather_point_id": str(point_id),
                        "start_date": str(row["start_date"]),
                        "end_date": str(row["end_date"]),
                        "weather_days_expected": int(row["expected_days"]),
                        "weather_days_present": 0,
                        "weather_coverage_ratio": 0.0,
                        "weather_window_valid": False,
                        "status": "error",
                        "error": str(exc),
                        **{f"missing_{param}": 0 for param in NASA_PARAMETERS},
                    }
                )
            continue

        for window in point_windows.itertuples(index=False):
            row = window._asdict()
            window_id = str(row["weather_window_id"])
            try:
                features, missing_counts = _aggregate_from_arrays(arrays, row["start_date"], row["end_date"])
                expected_from_input = int(row["expected_days"])
                if features["weather_days_expected"] != expected_from_input:
                    raise ValueError(
                        f"expected_days mismatch for {window_id}: "
                        f"{features['weather_days_expected']} != {expected_from_input}"
                    )
                base = {
                    "weather_window_id": window_id,
                    "weather_point_id": str(point_id),
                    "start_date": str(pd.to_datetime(row["start_date"]).date()),
                    "end_date": str(pd.to_datetime(row["end_date"]).date()),
                }
                feature_rows.append({**base, **features})
                status_rows.append(
                    {
                        **base,
                        "weather_days_expected": features["weather_days_expected"],
                        "weather_days_present": features["weather_days_present"],
                        "weather_coverage_ratio": features["weather_coverage_ratio"],
                        "weather_window_valid": features["weather_window_valid"],
                        "status": "ok" if features["weather_window_valid"] else "invalid_coverage",
                        "error": "",
                        **missing_counts,
                    }
                )
                if not features["weather_window_valid"]:
                    error_rows.append(
                        {
                            **base,
                            "error_type": "invalid_coverage",
                            "message": f"coverage={features['weather_coverage_ratio']:.6f}",
                        }
                    )
            except Exception as exc:
                error_rows.append(
                    {
                        "weather_window_id": window_id,
                        "weather_point_id": str(point_id),
                        "start_date": str(row["start_date"]),
                        "end_date": str(row["end_date"]),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                status_rows.append(
                    {
                        "weather_window_id": window_id,
                        "weather_point_id": point_id,
                        "start_date": str(row["start_date"]),
                        "end_date": str(row["end_date"]),
                        "weather_days_expected": int(row["expected_days"]),
                        "weather_days_present": 0,
                        "weather_coverage_ratio": 0.0,
                        "weather_window_valid": False,
                        "status": "error",
                        "error": str(exc),
                        **{f"missing_{param}": 0 for param in NASA_PARAMETERS},
                    }
                )

    features = pd.DataFrame(feature_rows)
    status = pd.DataFrame(status_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=["weather_window_id", "weather_point_id", "start_date", "end_date", "error_type", "message"],
    )
    if not features.empty:
        features = features[["weather_window_id", "weather_point_id", "start_date", "end_date", *FEATURE_COLUMNS]]
    return features, status, errors


def write_reports(windows: pd.DataFrame, features: pd.DataFrame, status: pd.DataFrame, errors: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(STATUS_PATH, index=False, lineterminator="\n")
    errors.to_csv(ERRORS_PATH, index=False, lineterminator="\n")

    if status.empty:
        coverage_min = coverage_mean = coverage_median = 0.0
        valid_windows = invalid_windows = 0
    else:
        coverage_min = float(status["weather_coverage_ratio"].min())
        coverage_mean = float(status["weather_coverage_ratio"].mean())
        coverage_median = float(status["weather_coverage_ratio"].median())
        valid_windows = int(status["weather_window_valid"].sum())
        invalid_windows = int((~status["weather_window_valid"].astype(bool)).sum())

    missing_lines = []
    for param in NASA_PARAMETERS:
        column = f"missing_{param}"
        count = int(status[column].sum()) if column in status.columns else 0
        missing_lines.append(f"- Missing {param}: {count}")

    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    lines = [
        "# Weather Window Aggregation Summary",
        "",
        f"- Input weather windows: {len(windows)}",
        f"- Feature rows created: {len(features)}",
        f"- Valid windows: {valid_windows}",
        f"- Invalid windows: {invalid_windows}",
        f"- Minimum coverage: {coverage_min:.6f}",
        f"- Mean coverage: {coverage_mean:.6f}",
        f"- Median coverage: {coverage_median:.6f}",
        f"- Start date range: {windows['start_date'].min()} to {windows['start_date'].max()}",
        f"- End date range: {windows['end_date'].min()} to {windows['end_date'].max()}",
        f"- Output parquet: {output_path.relative_to(REPO_ROOT)}",
        f"- Output parquet size bytes: {size_bytes}",
        "",
        "## Missing NASA Values",
        "",
        *missing_lines,
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    windows = pd.read_parquet(WINDOWS_PATH)
    features, status, errors = aggregate_windows(windows, CACHE_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    write_reports(windows, features, status, errors, OUTPUT_PATH)

    invalid_windows = int((~status["weather_window_valid"].astype(bool)).sum()) if not status.empty else 0
    print(f"weather_windows={len(windows)}")
    print(f"feature_rows={len(features)}")
    print(f"valid_windows={len(status) - invalid_windows}")
    print(f"invalid_windows={invalid_windows}")
    print(f"min_coverage={float(status['weather_coverage_ratio'].min()) if not status.empty else 0.0:.6f}")
    print(f"mean_coverage={float(status['weather_coverage_ratio'].mean()) if not status.empty else 0.0:.6f}")
    print(f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}")
    if not errors.empty:
        raise SystemExit("Weather aggregation found invalid windows or errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
