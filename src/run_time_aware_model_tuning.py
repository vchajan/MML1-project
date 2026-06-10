from __future__ import annotations

import argparse
import inspect
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import run_validation_benchmark as validation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor


REPO_ROOT = validation.REPO_ROOT
TRAIN_PATH = validation.TRAIN_PATH
VALIDATION_PATH = validation.VALIDATION_PATH
TEST_PATH = validation.TEST_PATH
FEATURE_MANIFEST_PATH = validation.FEATURE_MANIFEST_PATH

RUNS_PATH = REPO_ROOT / "data" / "interim" / "time_aware_tuning_runs.csv"
MODELS_DIR = REPO_ROOT / "data" / "interim" / "time_aware_tuning_models"
PREDICTIONS_DIR = REPO_ROOT / "data" / "interim" / "time_aware_tuning_predictions"

SHORTLIST_PATH = REPO_ROOT / "data" / "reference" / "time_cv_shortlist.json"
FROZEN_TUNED_CONFIG_PATH = REPO_ROOT / "data" / "reference" / "frozen_tuned_model_configuration.json"

TIME_CV_BASELINE_FOLD_RESULTS_PATH = REPO_ROOT / "reports" / "time_cv_baseline_fold_results.csv"
TIME_CV_DIRECT_RESULTS_PATH = REPO_ROOT / "reports" / "time_cv_direct_results.csv"
TIME_CV_RESIDUAL_RESULTS_PATH = REPO_ROOT / "reports" / "time_cv_residual_results.csv"
TIME_CV_LOG_TARGET_RESULTS_PATH = REPO_ROOT / "reports" / "time_cv_log_target_results.csv"
TIME_CV_ALL_RESULTS_PATH = REPO_ROOT / "reports" / "time_cv_all_results.csv"
TUNED_VALIDATION_RESULTS_PATH = REPO_ROOT / "reports" / "tuned_validation_results.csv"
TUNED_VALIDATION_RUNTIME_PATH = REPO_ROOT / "reports" / "tuned_validation_runtime.csv"
TUNED_VALIDATION_PREDICTIONS_SAMPLE_PATH = REPO_ROOT / "reports" / "tuned_validation_predictions_sample.csv"
TUNED_VALIDATION_COMPARISON_PATH = REPO_ROOT / "reports" / "tuned_validation_comparison.csv"
TUNED_VALIDATION_SUBGROUP_METRICS_PATH = REPO_ROOT / "reports" / "tuned_validation_subgroup_metrics.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "time_aware_tuning_summary.md"
TIME_CV_MAE_PLOT_PATH = REPO_ROOT / "reports" / "time_cv_mae.png"
TIME_CV_STABILITY_PLOT_PATH = REPO_ROOT / "reports" / "time_cv_stability.png"
TUNED_VALIDATION_MAE_PLOT_PATH = REPO_ROOT / "reports" / "tuned_validation_mae.png"

TARGET_COLUMN = validation.TARGET_COLUMN
TRAIN_YEARS = set(range(1997, 2011))
VALIDATION_YEARS = {2011, 2012}
EXPECTED_TRAIN_ROWS = 202_164
EXPECTED_VALIDATION_ROWS = 32_388
TEST_DATA_ACCESSED = False
ROLLING_LAG_ASSUMPTION = (
    "Previous-year official yield is assumed to be available when predicting the following year."
)
FORECAST_TRACK = "forecast_with_lag"
SUITABILITY_TRACK = "suitability_without_lag"
MAX_STABLE_LOG_TARGET_WORST_RMSE = 20.0
FROZEN_TRACK_RUN_IDS = {
    "forecast_baseline": "cv_baseline_lag_with_crop_median_fallback",
    "forecast_trained_model": "cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector",
    "suitability_model": "cv_direct_tree_depth_none_leaf_20_core_without_lag",
    "log_target_experiment": "cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag",
}
FROZEN_TRACK_MODEL_NAMES = {
    "LinearSVR": "LinearSVR",
    "DecisionTree": "DecisionTreeRegressor",
    "RandomForest": "RandomForestRegressor",
}

FOLD_IDS = ("fold_1", "fold_2", "fold_3")

RUN_COLUMNS = [
    "model_run_id",
    "phase",
    "application_track",
    "experiment_family",
    "model_name",
    "model_family",
    "feature_set",
    "target_strategy",
    "fold_id",
    "fit_start_year",
    "fit_end_year",
    "evaluation_start_year",
    "evaluation_end_year",
    "hyperparameters_json",
    "preprocessing_family",
    "training_scope",
    "fit_rows",
    "evaluation_rows",
    "fit_seconds",
    "predict_seconds",
    "mae",
    "rmse",
    "r2",
    "median_ae",
    "lag_subset_rows",
    "lag_subset_mae",
    "lag_subset_rmse",
    "baseline_fold_mae",
    "mae_improvement_over_lag_baseline",
    "time_cv_mean_mae",
    "time_cv_std_mae",
    "time_cv_worst_fold_mae",
    "status",
    "warning_count",
    "warning_summary",
    "error_type",
    "error_message",
    "random_state",
    "test_data_accessed",
]

MODEL_SIMPLICITY = {
    "GlobalMedian": 0,
    "CropMedian": 1,
    "LagWithCropMedianFallback": 2,
    "Ridge": 3,
    "LinearSVR": 4,
    "DecisionTree": 5,
    "RandomForest": 6,
}


@dataclass(frozen=True)
class TimeFold:
    fold_id: str
    fit_start_year: int
    fit_end_year: int
    evaluation_start_year: int
    evaluation_end_year: int


@dataclass(frozen=True)
class TuningConfig:
    model_run_id: str
    experiment_family: str
    model_name: str
    model_family: str
    feature_set: str
    target_strategy: str
    preprocessing_family: str
    hyperparameters: dict[str, Any]
    training_scope: str = "full_train"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def sanitize(value: Any) -> Any:
    return validation.sanitize_for_json(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    validation.write_json(path, value)


def value_slug(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).lower().replace(".", "_").replace("-", "neg_")
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def get_time_cv_folds() -> list[TimeFold]:
    return [
        TimeFold("fold_1", 1997, 2004, 2005, 2006),
        TimeFold("fold_2", 1997, 2006, 2007, 2008),
        TimeFold("fold_3", 1997, 2008, 2009, 2010),
    ]


def rows_for_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    validation.require_columns(frame, ["Crop_Year"], "year-filter frame")
    years = frame["Crop_Year"].astype(int)
    return frame[years.between(start_year, end_year)].copy()


def rows_for_fold(frame: pd.DataFrame, fold: TimeFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold.fit_end_year >= fold.evaluation_start_year:
        raise ValueError("fold fit years must end before evaluation years begin")
    fit = rows_for_years(frame, fold.fit_start_year, fold.fit_end_year)
    evaluation = rows_for_years(frame, fold.evaluation_start_year, fold.evaluation_end_year)
    return fit, evaluation


def validate_time_cv_folds(folds: Iterable[TimeFold]) -> None:
    for fold in folds:
        if fold.fit_start_year > fold.fit_end_year:
            raise ValueError(f"{fold.fold_id} has an empty fit period")
        if fold.fit_end_year >= fold.evaluation_start_year:
            raise ValueError(f"{fold.fold_id} fit years overlap evaluation years")


def guard_not_test_path(path: Path) -> None:
    validation.guard_not_test_path(path)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    guard_not_test_path(path)
    return pd.read_parquet(path)


def load_train_only(
    train_path: Path = TRAIN_PATH,
    manifest_path: Path = FEATURE_MANIFEST_PATH,
    expected_rows: int | None = EXPECTED_TRAIN_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = safe_read_parquet(train_path)
    manifest = validation.load_manifest(manifest_path)
    if expected_rows is not None and len(train) != expected_rows:
        raise ValueError(f"Expected {expected_rows} train rows, found {len(train)}")
    validation.validate_years(train, TRAIN_YEARS, "train")
    validation.validate_target(train, "train")
    for name, features in manifest.get("feature_sets", {}).items():
        validate_feature_columns(name, list(features), train.columns, manifest)
    return train, manifest


def load_train_validation(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    manifest_path: Path = FEATURE_MANIFEST_PATH,
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS,
    expected_validation_rows: int | None = EXPECTED_VALIDATION_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = safe_read_parquet(train_path)
    validation_frame = safe_read_parquet(validation_path)
    manifest = validation.load_manifest(manifest_path)
    validation.validate_benchmark_inputs(
        train,
        validation_frame,
        manifest,
        expected_train_rows=expected_train_rows,
        expected_validation_rows=expected_validation_rows,
    )
    return train, validation_frame, manifest


def fit_crop_medians(
    frame: pd.DataFrame,
    crop_column: str = "Crop_canonical",
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    validation.require_columns(frame, [crop_column, target_column], "crop-median fit data")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError("crop-median target contains missing values")
    return {
        "crop_column": crop_column,
        "global_median": float(target.median()),
        "crop_medians": target.groupby(frame[crop_column]).median().to_dict(),
    }


def predict_crop_median(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    crop_column = str(state.get("crop_column", "Crop_canonical"))
    validation.require_columns(frame, [crop_column], "crop-median prediction data")
    predictions = frame[crop_column].map(state["crop_medians"]).fillna(float(state["global_median"]))
    return predictions.astype("float64").to_numpy()


def predict_lag_with_crop_median_fallback(frame: pd.DataFrame, crop_median_state: dict[str, Any]) -> np.ndarray:
    required = ["Crop_canonical", "lag_available", "lag_yield_1y"]
    validation.require_columns(frame, required, "lag baseline data")
    predictions = pd.Series(predict_crop_median(frame, crop_median_state), index=frame.index)
    lag_values = pd.to_numeric(frame["lag_yield_1y"], errors="coerce")
    use_lag = frame["lag_available"].astype(int).eq(1) & lag_values.notna()
    predictions.loc[use_lag] = lag_values.loc[use_lag]
    return predictions.astype("float64").to_numpy()


def residual_fit_mask(frame: pd.DataFrame) -> pd.Series:
    validation.require_columns(frame, ["lag_available", "lag_yield_1y"], "residual fit data")
    lag_values = pd.to_numeric(frame["lag_yield_1y"], errors="coerce")
    return frame["lag_available"].astype(int).eq(1) & lag_values.notna()


def compute_residual_target(frame: pd.DataFrame, baseline_predictions: Iterable[float]) -> np.ndarray:
    validation.require_columns(frame, [TARGET_COLUMN], "residual target data")
    return frame[TARGET_COLUMN].astype("float64").to_numpy() - np.asarray(list(baseline_predictions), dtype="float64")


def apply_residual_predictions(
    frame: pd.DataFrame,
    baseline_predictions: Iterable[float],
    residual_predictions: Iterable[float],
) -> np.ndarray:
    baseline = np.asarray(list(baseline_predictions), dtype="float64").copy()
    mask = residual_fit_mask(frame).to_numpy()
    residuals = np.asarray(list(residual_predictions), dtype="float64")
    if residuals.shape[0] != int(mask.sum()):
        raise ValueError("Residual prediction count does not match rows with available lag")
    baseline[mask] = baseline[mask] + residuals
    return baseline


def inverse_log_predictions(log_predictions: Iterable[float]) -> np.ndarray:
    raw = np.expm1(np.asarray(list(log_predictions), dtype="float64"))
    return np.maximum(raw, 0.0)


def validate_feature_columns(
    feature_set_name: str,
    features: list[str],
    frame_columns: Iterable[str],
    manifest: dict[str, Any],
) -> None:
    if TARGET_COLUMN in features:
        raise ValueError(f"{feature_set_name} includes target_yield")
    forbidden = set(manifest.get("forbidden_leakage_columns", []))
    forbidden_used = sorted(forbidden.intersection(features))
    if forbidden_used:
        raise ValueError(f"{feature_set_name} contains forbidden leakage columns: {', '.join(forbidden_used)}")
    missing = sorted(set(features) - set(frame_columns))
    if missing:
        raise ValueError(f"{feature_set_name} is missing approved features: {', '.join(missing)}")


def features_for_config(config: TuningConfig, manifest: dict[str, Any]) -> list[str]:
    if config.target_strategy == "baseline":
        return []
    if config.target_strategy == "residual":
        features = list(manifest["feature_sets"]["core_without_lag"])
        for lag_feature in ["lag_yield_1y", "lag_available"]:
            if lag_feature not in features:
                features.append(lag_feature)
        return features
    return list(manifest["feature_sets"][config.feature_set])


def build_preprocessor(config: TuningConfig, features: list[str], manifest: dict[str, Any]) -> Any:
    if config.preprocessing_family == "linear":
        return validation.make_linear_preprocessor(features, manifest)
    if config.preprocessing_family == "tree":
        return validation.make_tree_preprocessor(features, manifest)
    if config.preprocessing_family == "none":
        return None
    raise ValueError(f"Unsupported preprocessing family: {config.preprocessing_family}")


def make_linearsvr_from_config(hyperparameters: dict[str, Any], random_state: int) -> LinearSVR:
    params = dict(hyperparameters)
    params.setdefault("max_iter", 20_000)
    params.setdefault("random_state", random_state)
    if "dual" not in params and validation.sklearn_version_tuple() >= (1, 3):
        params["dual"] = "auto"
    return LinearSVR(**params)


def make_estimator(config: TuningConfig, random_state: int) -> Any:
    params = dict(config.hyperparameters)
    if config.model_name == "Ridge":
        return Ridge(**params)
    if config.model_name == "LinearSVR":
        return make_linearsvr_from_config(params, random_state)
    if config.model_name == "DecisionTree":
        params.setdefault("random_state", random_state)
        return DecisionTreeRegressor(**params)
    if config.model_name == "RandomForest":
        params.setdefault("random_state", random_state)
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported model: {config.model_name}")


def make_model_pipeline(
    config: TuningConfig,
    features: list[str],
    manifest: dict[str, Any],
    random_state: int,
) -> Pipeline:
    preprocessor = build_preprocessor(config, features, manifest)
    model = make_estimator(config, random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    return validation.compute_regression_metrics(y_true, y_pred)


def lag_subset_metrics(frame: pd.DataFrame, predictions: Iterable[float]) -> tuple[int, float, float]:
    if "lag_available" not in frame.columns:
        return 0, np.nan, np.nan
    mask = frame["lag_available"].astype(int).eq(1)
    rows = int(mask.sum())
    if rows == 0:
        return 0, np.nan, np.nan
    pred = np.asarray(list(predictions), dtype="float64")
    metrics = regression_metrics(frame.loc[mask, TARGET_COLUMN], pred[mask.to_numpy()])
    return rows, metrics["mae"], metrics["rmse"]


def baseline_predictions_for_config(
    config: TuningConfig,
    fit_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    if config.model_name == "GlobalMedian":
        median = float(pd.to_numeric(fit_frame[TARGET_COLUMN], errors="coerce").median())
        return np.repeat(median, len(evaluation_frame)).astype("float64"), len(fit_frame)
    crop_state = fit_crop_medians(fit_frame)
    if config.model_name == "CropMedian":
        return predict_crop_median(evaluation_frame, crop_state), len(fit_frame)
    if config.model_name == "LagWithCropMedianFallback":
        return predict_lag_with_crop_median_fallback(evaluation_frame, crop_state), len(fit_frame)
    raise ValueError(f"Unsupported baseline model: {config.model_name}")


def lag_baseline_predictions(fit_frame: pd.DataFrame, evaluation_frame: pd.DataFrame) -> np.ndarray:
    crop_state = fit_crop_medians(fit_frame)
    return predict_lag_with_crop_median_fallback(evaluation_frame, crop_state)


def empty_result(
    config: TuningConfig,
    phase: str,
    fold_id: str,
    fit_start_year: int,
    fit_end_year: int,
    evaluation_start_year: int,
    evaluation_end_year: int,
    fit_rows: int,
    evaluation_rows: int,
    random_state: int,
) -> dict[str, Any]:
    return {
        "model_run_id": config.model_run_id,
        "phase": phase,
        "application_track": "",
        "experiment_family": config.experiment_family,
        "model_name": config.model_name,
        "model_family": config.model_family,
        "feature_set": config.feature_set,
        "target_strategy": config.target_strategy,
        "fold_id": fold_id,
        "fit_start_year": fit_start_year,
        "fit_end_year": fit_end_year,
        "evaluation_start_year": evaluation_start_year,
        "evaluation_end_year": evaluation_end_year,
        "hyperparameters_json": json_dumps(config.hyperparameters),
        "preprocessing_family": config.preprocessing_family,
        "training_scope": config.training_scope,
        "fit_rows": fit_rows,
        "evaluation_rows": evaluation_rows,
        "fit_seconds": np.nan,
        "predict_seconds": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "median_ae": np.nan,
        "lag_subset_rows": 0,
        "lag_subset_mae": np.nan,
        "lag_subset_rmse": np.nan,
        "baseline_fold_mae": np.nan,
        "mae_improvement_over_lag_baseline": np.nan,
        "time_cv_mean_mae": np.nan,
        "time_cv_std_mae": np.nan,
        "time_cv_worst_fold_mae": np.nan,
        "status": "failed",
        "warning_count": 0,
        "warning_summary": "",
        "error_type": "",
        "error_message": "",
        "random_state": random_state,
        "test_data_accessed": False,
    }


def prediction_path(model_run_id: str, phase: str = "validation-shortlist") -> Path:
    return PREDICTIONS_DIR / f"{model_run_id}_{phase}.csv"


def write_predictions(
    model_run_id: str,
    phase: str,
    evaluation_frame: pd.DataFrame,
    predictions: Iterable[float],
) -> None:
    ensure_parent(prediction_path(model_run_id, phase))
    frame = pd.DataFrame(
        {
            "canonical_crop_row_id": evaluation_frame["canonical_crop_row_id"].astype(str),
            "Crop_Year": evaluation_frame["Crop_Year"].astype(int),
            "canonical_state_name": evaluation_frame.get("canonical_state_name", "").astype(str),
            "canonical_district_name": evaluation_frame.get("canonical_district_name", "").astype(str),
            "Crop_canonical": evaluation_frame.get("Crop_canonical", "").astype(str),
            "Season_canonical": evaluation_frame.get("Season_canonical", "").astype(str),
            "target_yield": evaluation_frame[TARGET_COLUMN].astype("float64"),
            "prediction": np.asarray(list(predictions), dtype="float64"),
        }
    )
    frame["absolute_error"] = (frame["target_yield"] - frame["prediction"]).abs()
    frame.to_csv(prediction_path(model_run_id, phase), index=False)


def execute_on_split(
    config: TuningConfig,
    fit_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    manifest: dict[str, Any],
    random_state: int,
    phase: str,
    fold_id: str,
    fit_start_year: int,
    fit_end_year: int,
    evaluation_start_year: int,
    evaluation_end_year: int,
    cv_metrics: dict[str, Any] | None = None,
    write_prediction_file: bool = False,
) -> dict[str, Any]:
    result = empty_result(
        config,
        phase,
        fold_id,
        fit_start_year,
        fit_end_year,
        evaluation_start_year,
        evaluation_end_year,
        fit_rows=len(fit_frame),
        evaluation_rows=len(evaluation_frame),
        random_state=random_state,
    )
    if cv_metrics:
        result["time_cv_mean_mae"] = cv_metrics.get("mean_mae", np.nan)
        result["time_cv_std_mae"] = cv_metrics.get("std_mae", np.nan)
        result["time_cv_worst_fold_mae"] = cv_metrics.get("worst_fold_mae", np.nan)

    try:
        validation.validate_target(fit_frame, "fit")
        validation.validate_target(evaluation_frame, "evaluation")
        lag_baseline = lag_baseline_predictions(fit_frame, evaluation_frame)
        lag_baseline_mae = regression_metrics(evaluation_frame[TARGET_COLUMN], lag_baseline)["mae"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_start = time.perf_counter()
            if config.target_strategy == "baseline":
                predictions, fit_rows_used = baseline_predictions_for_config(config, fit_frame, evaluation_frame)
                fit_seconds = time.perf_counter() - fit_start
                predict_seconds = 0.0
            elif config.target_strategy in {"direct", "log_target"}:
                features = features_for_config(config, manifest)
                validate_feature_columns(config.feature_set, features, fit_frame.columns, manifest)
                validate_feature_columns(config.feature_set, features, evaluation_frame.columns, manifest)
                estimator = make_model_pipeline(config, features, manifest, random_state)
                y_fit = fit_frame[TARGET_COLUMN].astype("float64")
                if config.target_strategy == "log_target":
                    y_fit = np.log1p(y_fit)
                estimator.fit(fit_frame[features], y_fit)
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                raw_predictions = estimator.predict(evaluation_frame[features])
                predict_seconds = time.perf_counter() - predict_start
                predictions = (
                    inverse_log_predictions(raw_predictions)
                    if config.target_strategy == "log_target"
                    else np.asarray(raw_predictions, dtype="float64")
                )
                fit_rows_used = len(fit_frame)
            elif config.target_strategy == "residual":
                features = features_for_config(config, manifest)
                validate_feature_columns(config.feature_set, features, fit_frame.columns, manifest)
                validate_feature_columns(config.feature_set, features, evaluation_frame.columns, manifest)
                fit_baseline = lag_baseline_predictions(fit_frame, fit_frame)
                residual_target = compute_residual_target(fit_frame, fit_baseline)
                train_mask = residual_fit_mask(fit_frame)
                if int(train_mask.sum()) == 0:
                    raise ValueError("Residual model has no lag-available fit rows")
                estimator = make_model_pipeline(config, features, manifest, random_state)
                estimator.fit(fit_frame.loc[train_mask, features], residual_target[train_mask.to_numpy()])
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                eval_mask = residual_fit_mask(evaluation_frame)
                residual_predictions = estimator.predict(evaluation_frame.loc[eval_mask, features])
                predict_seconds = time.perf_counter() - predict_start
                predictions = apply_residual_predictions(evaluation_frame, lag_baseline, residual_predictions)
                fit_rows_used = int(train_mask.sum())
            else:
                raise ValueError(f"Unsupported target strategy: {config.target_strategy}")

        metrics = regression_metrics(evaluation_frame[TARGET_COLUMN], predictions)
        lag_rows, lag_mae, lag_rmse = lag_subset_metrics(evaluation_frame, predictions)
        warning_messages = sorted({str(item.message) for item in caught})
        result.update(metrics)
        result.update(
            {
                "fit_rows": fit_rows_used,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "lag_subset_rows": lag_rows,
                "lag_subset_mae": lag_mae,
                "lag_subset_rmse": lag_rmse,
                "baseline_fold_mae": lag_baseline_mae,
                "mae_improvement_over_lag_baseline": lag_baseline_mae - metrics["mae"],
                "status": "completed",
                "warning_count": len(caught),
                "warning_summary": " | ".join(message[:250] for message in warning_messages),
                "test_data_accessed": False,
            }
        )
        if write_prediction_file:
            write_predictions(config.model_run_id, phase, evaluation_frame, predictions)
    except MemoryError as error:
        result.update({"status": "failed", "error_type": "MemoryError", "error_message": str(error)})
    except Exception as error:
        result.update({"status": "failed", "error_type": type(error).__name__, "error_message": str(error)})
    return result


def execute_cv_fold(
    config: TuningConfig,
    train: pd.DataFrame,
    manifest: dict[str, Any],
    fold: TimeFold,
    random_state: int,
) -> dict[str, Any]:
    fit_frame, evaluation_frame = rows_for_fold(train, fold)
    return execute_on_split(
        config,
        fit_frame,
        evaluation_frame,
        manifest,
        random_state=random_state,
        phase=f"cv-{config.experiment_family}",
        fold_id=fold.fold_id,
        fit_start_year=fold.fit_start_year,
        fit_end_year=fold.fit_end_year,
        evaluation_start_year=fold.evaluation_start_year,
        evaluation_end_year=fold.evaluation_end_year,
    )


def execute_validation_config(
    config: TuningConfig,
    train: pd.DataFrame,
    validation_frame: pd.DataFrame,
    manifest: dict[str, Any],
    random_state: int,
    cv_metrics: dict[str, Any] | None = None,
    write_prediction_file: bool = True,
) -> dict[str, Any]:
    return execute_on_split(
        config,
        train,
        validation_frame,
        manifest,
        random_state=random_state,
        phase="validation-shortlist",
        fold_id="validation_2011_2012",
        fit_start_year=1997,
        fit_end_year=2010,
        evaluation_start_year=2011,
        evaluation_end_year=2012,
        cv_metrics=cv_metrics,
        write_prediction_file=write_prediction_file,
    )


def read_runs(path: Path = RUNS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RUN_COLUMNS)
    frame = pd.read_csv(path)
    for column in RUN_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[RUN_COLUMNS]


def write_run_result(result: dict[str, Any], path: Path = RUNS_PATH) -> None:
    ensure_parent(path)
    existing = read_runs(path)
    same = (
        existing["model_run_id"].astype(str).eq(str(result["model_run_id"]))
        & existing["fold_id"].astype(str).eq(str(result["fold_id"]))
        & existing["phase"].astype(str).eq(str(result["phase"]))
    )
    existing = existing[~same]
    new_row = pd.DataFrame([{column: result.get(column, np.nan) for column in RUN_COLUMNS}])
    pd.concat([existing, new_row], ignore_index=True).to_csv(path, index=False)


def has_completed_run(existing_runs: pd.DataFrame, model_run_id: str, fold_id: str, phase: str) -> bool:
    if existing_runs.empty:
        return False
    matches = existing_runs[
        existing_runs["model_run_id"].astype(str).eq(model_run_id)
        & existing_runs["fold_id"].astype(str).eq(fold_id)
        & existing_runs["phase"].astype(str).eq(phase)
        & existing_runs["status"].astype(str).eq("completed")
    ]
    return not matches.empty


def should_skip_run(
    config: TuningConfig,
    existing_runs: pd.DataFrame,
    resume: bool,
    fold_id: str,
    phase: str,
    force_run: str | None = None,
) -> bool:
    if force_run is not None and force_run != config.model_run_id:
        return True
    if force_run is not None:
        return False
    return resume and has_completed_run(existing_runs, config.model_run_id, fold_id, phase)


def run_config_sequence(
    configs: list[TuningConfig],
    folds: list[TimeFold],
    train: pd.DataFrame,
    manifest: dict[str, Any],
    resume: bool,
    force_run: str | None,
    random_state: int,
    evaluator: Callable[[TuningConfig, pd.DataFrame, dict[str, Any], TimeFold, int], dict[str, Any]] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    evaluator = evaluator or execute_cv_fold
    writer = writer or write_run_result
    results: list[dict[str, Any]] = []
    for config in configs:
        phase = f"cv-{config.experiment_family}"
        for fold in folds:
            existing_runs = read_runs()
            if should_skip_run(config, existing_runs, resume, fold.fold_id, phase, force_run):
                print(f"skipped_completed={config.model_run_id} fold={fold.fold_id}")
                continue
            try:
                result = evaluator(config, train, manifest, fold, random_state)
            except MemoryError as error:
                result = empty_result(
                    config,
                    phase,
                    fold.fold_id,
                    fold.fit_start_year,
                    fold.fit_end_year,
                    fold.evaluation_start_year,
                    fold.evaluation_end_year,
                    0,
                    0,
                    random_state,
                )
                result.update({"error_type": "MemoryError", "error_message": str(error)})
            except Exception as error:
                result = empty_result(
                    config,
                    phase,
                    fold.fold_id,
                    fold.fit_start_year,
                    fold.fit_end_year,
                    fold.evaluation_start_year,
                    fold.evaluation_end_year,
                    0,
                    0,
                    random_state,
                )
                result.update({"error_type": type(error).__name__, "error_message": str(error)})
            writer(result)
            print(
                f"model_run_id={result['model_run_id']} fold={result['fold_id']} "
                f"status={result['status']} mae={result.get('mae', np.nan)} rmse={result.get('rmse', np.nan)}"
            )
            results.append(result)
    return results


def baseline_configs() -> list[TuningConfig]:
    return [
        TuningConfig(
            model_run_id="cv_baseline_global_median",
            experiment_family="baseline",
            model_name="GlobalMedian",
            model_family="baseline",
            feature_set="none",
            target_strategy="baseline",
            preprocessing_family="none",
            hyperparameters={"strategy": "median"},
        ),
        TuningConfig(
            model_run_id="cv_baseline_crop_median",
            experiment_family="baseline",
            model_name="CropMedian",
            model_family="baseline",
            feature_set="crop_only",
            target_strategy="baseline",
            preprocessing_family="none",
            hyperparameters={"fallback": "global_fit_median"},
        ),
        TuningConfig(
            model_run_id="cv_baseline_lag_with_crop_median_fallback",
            experiment_family="baseline",
            model_name="LagWithCropMedianFallback",
            model_family="baseline",
            feature_set="lag_with_crop_fallback",
            target_strategy="baseline",
            preprocessing_family="none",
            hyperparameters={"fallback": "crop_fit_median"},
        ),
    ]


def ridge_config(prefix: str, family: str, feature_set: str, target_strategy: str, alpha: float) -> TuningConfig:
    run_id = f"{prefix}_ridge_alpha_{value_slug(alpha)}_{feature_set}"
    return TuningConfig(
        model_run_id=run_id,
        experiment_family=family,
        model_name="Ridge",
        model_family="linear",
        feature_set=feature_set,
        target_strategy=target_strategy,
        preprocessing_family="linear",
        hyperparameters={"alpha": alpha, "solver": "auto"},
    )


def linearsvr_config(
    prefix: str,
    family: str,
    feature_set: str,
    target_strategy: str,
    c_value: float,
    epsilon: float,
) -> TuningConfig:
    run_id = f"{prefix}_linearsvr_c_{value_slug(c_value)}_epsilon_{value_slug(epsilon)}_{feature_set}"
    return TuningConfig(
        model_run_id=run_id,
        experiment_family=family,
        model_name="LinearSVR",
        model_family="linear",
        feature_set=feature_set,
        target_strategy=target_strategy,
        preprocessing_family="linear",
        hyperparameters={"C": c_value, "epsilon": epsilon, "max_iter": 20_000},
    )


def tree_config(prefix: str, feature_set: str, max_depth: int | None, min_samples_leaf: int) -> TuningConfig:
    run_id = f"{prefix}_tree_depth_{value_slug(max_depth)}_leaf_{value_slug(min_samples_leaf)}_{feature_set}"
    return TuningConfig(
        model_run_id=run_id,
        experiment_family="direct",
        model_name="DecisionTree",
        model_family="tree",
        feature_set=feature_set,
        target_strategy="direct",
        preprocessing_family="tree",
        hyperparameters={"max_depth": max_depth, "min_samples_leaf": min_samples_leaf},
    )


def forest_config(
    prefix: str,
    family: str,
    feature_set: str,
    target_strategy: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
) -> TuningConfig:
    run_id = (
        f"{prefix}_rf_{n_estimators}_depth_{value_slug(max_depth)}_"
        f"leaf_{min_samples_leaf}_maxfeat_{value_slug(max_features)}_{feature_set}"
    )
    return TuningConfig(
        model_run_id=run_id,
        experiment_family=family,
        model_name="RandomForest",
        model_family="tree",
        feature_set=feature_set,
        target_strategy=target_strategy,
        preprocessing_family="tree",
        hyperparameters={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_jobs": -1,
        },
    )


def direct_configs(random_state: int = 42) -> list[TuningConfig]:
    configs: list[TuningConfig] = []
    for feature_set in ["core_without_lag", "core_with_lag"]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            configs.append(ridge_config("cv_direct", "direct", feature_set, "direct", alpha))
        for c_value in [0.03, 0.1, 0.3, 1.0]:
            for epsilon in [0.0, 0.1]:
                configs.append(linearsvr_config("cv_direct", "direct", feature_set, "direct", c_value, epsilon))
        for max_depth, min_samples_leaf in [(8, 20), (12, 20), (16, 10), (None, 20), (None, 50)]:
            configs.append(tree_config("cv_direct", feature_set, max_depth, min_samples_leaf))
        forest_specs = [
            (150, 15, 5, "sqrt"),
            (200, 25, 5, "sqrt"),
            (250, None, 10, "sqrt"),
            (200, None, 20, 0.5),
        ]
        for spec in forest_specs:
            configs.append(forest_config("cv_direct", "direct", feature_set, "direct", *spec))
    return configs


def residual_configs(random_state: int = 42) -> list[TuningConfig]:
    feature_set = "residual_lag_corrector"
    configs: list[TuningConfig] = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        configs.append(ridge_config("cv_residual", "residual", feature_set, "residual", alpha))
    for c_value in [0.03, 0.1, 0.3, 1.0]:
        configs.append(linearsvr_config("cv_residual", "residual", feature_set, "residual", c_value, 0.0))
    forest_specs = [
        (200, 20, 10, "sqrt"),
        (250, None, 20, "sqrt"),
    ]
    for spec in forest_specs:
        configs.append(forest_config("cv_residual", "residual", feature_set, "residual", *spec))
    return configs


def log_target_configs(random_state: int = 42) -> list[TuningConfig]:
    feature_set = "core_with_lag"
    configs: list[TuningConfig] = []
    for alpha in [1.0, 10.0]:
        configs.append(ridge_config("cv_log_target", "log_target", feature_set, "log_target", alpha))
    for c_value in [0.1, 0.3]:
        for epsilon in [0.0, 0.1]:
            configs.append(linearsvr_config("cv_log_target", "log_target", feature_set, "log_target", c_value, epsilon))
    forest_specs = [
        (200, 20, 10, "sqrt"),
        (250, None, 20, "sqrt"),
    ]
    for spec in forest_specs:
        configs.append(forest_config("cv_log_target", "log_target", feature_set, "log_target", *spec))
    return configs


def all_config_registry(random_state: int = 42) -> dict[str, TuningConfig]:
    configs = baseline_configs() + direct_configs(random_state) + residual_configs(random_state) + log_target_configs(random_state)
    return {config.model_run_id: config for config in configs}


def successful_cv_rows(experiment_family: str, runs_path: Path = RUNS_PATH) -> pd.DataFrame:
    runs = read_runs(runs_path)
    return runs[
        runs["experiment_family"].astype(str).eq(experiment_family)
        & runs["phase"].astype(str).str.startswith("cv-")
        & runs["status"].astype(str).eq("completed")
    ].copy()


def aggregate_cv_results(experiment_family: str, runs_path: Path = RUNS_PATH) -> pd.DataFrame:
    runs = read_runs(runs_path)
    family_rows = runs[
        runs["experiment_family"].astype(str).eq(experiment_family)
        & runs["phase"].astype(str).str.startswith("cv-")
    ].copy()
    if family_rows.empty:
        return pd.DataFrame()
    completed = family_rows[family_rows["status"].astype(str).eq("completed")].copy()
    grouped = []
    for model_run_id, group in completed.groupby("model_run_id", sort=False):
        failed = family_rows[
            family_rows["model_run_id"].astype(str).eq(str(model_run_id))
            & ~family_rows["status"].astype(str).eq("completed")
        ]
        first = group.iloc[0]
        grouped.append(
            {
                "model_run_id": model_run_id,
                "experiment_family": first["experiment_family"],
                "model_name": first["model_name"],
                "model_family": first["model_family"],
                "feature_set": first["feature_set"],
                "target_strategy": first["target_strategy"],
                "hyperparameters_json": first["hyperparameters_json"],
                "preprocessing_family": first["preprocessing_family"],
                "training_scope": first["training_scope"],
                "mean_mae": float(group["mae"].mean()),
                "std_mae": float(group["mae"].std(ddof=0)),
                "mean_rmse": float(group["rmse"].mean()),
                "std_rmse": float(group["rmse"].std(ddof=0)),
                "mean_r2": float(group["r2"].mean()),
                "std_r2": float(group["r2"].std(ddof=0)),
                "worst_fold_mae": float(group["mae"].max()),
                "worst_fold_rmse": float(group["rmse"].max()),
                "total_fit_seconds": float(group["fit_seconds"].sum()),
                "successful_folds": int(len(group)),
                "failed_folds": int(len(failed)),
                "simplicity_rank": MODEL_SIMPLICITY.get(str(first["model_name"]), 999),
                "test_data_accessed": False,
            }
        )
    result = pd.DataFrame(grouped)
    if not result.empty:
        result = result.sort_values(
            ["mean_mae", "worst_fold_mae", "std_mae", "mean_rmse", "simplicity_rank", "model_run_id"]
        )
    return result


def write_baseline_fold_report() -> pd.DataFrame:
    rows = successful_cv_rows("baseline")
    ensure_parent(TIME_CV_BASELINE_FOLD_RESULTS_PATH)
    rows.to_csv(TIME_CV_BASELINE_FOLD_RESULTS_PATH, index=False)
    return rows


def write_cv_aggregate_report(experiment_family: str) -> pd.DataFrame:
    path_by_family = {
        "direct": TIME_CV_DIRECT_RESULTS_PATH,
        "residual": TIME_CV_RESIDUAL_RESULTS_PATH,
        "log_target": TIME_CV_LOG_TARGET_RESULTS_PATH,
    }
    aggregate = aggregate_cv_results(experiment_family)
    ensure_parent(path_by_family[experiment_family])
    aggregate.to_csv(path_by_family[experiment_family], index=False)
    return aggregate


def load_or_build_aggregate(experiment_family: str) -> pd.DataFrame:
    path_by_family = {
        "baseline": None,
        "direct": TIME_CV_DIRECT_RESULTS_PATH,
        "residual": TIME_CV_RESIDUAL_RESULTS_PATH,
        "log_target": TIME_CV_LOG_TARGET_RESULTS_PATH,
    }
    if experiment_family == "baseline":
        return aggregate_cv_results("baseline")
    path = path_by_family[experiment_family]
    if path is not None and path.exists():
        frame = pd.read_csv(path)
        if "worst_fold_rmse" in frame.columns:
            return frame
    return write_cv_aggregate_report(experiment_family)


def rank_cv_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    if "simplicity_rank" not in ranked.columns:
        ranked["simplicity_rank"] = ranked["model_name"].map(MODEL_SIMPLICITY).fillna(999)
    return ranked.sort_values(
        ["mean_mae", "worst_fold_mae", "std_mae", "mean_rmse", "simplicity_rank", "model_run_id"]
    )


def finite_cv_metrics(frame: pd.DataFrame) -> pd.Series:
    required = ["mean_mae", "std_mae", "mean_rmse", "std_rmse", "mean_r2", "std_r2", "worst_fold_mae"]
    available = [column for column in required if column in frame.columns]
    if not available:
        return pd.Series(False, index=frame.index)
    return frame[available].apply(lambda row: np.isfinite(pd.to_numeric(row, errors="coerce")).all(), axis=1)


def stable_log_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    if "worst_fold_rmse" not in working.columns:
        working["worst_fold_rmse"] = working["mean_rmse"]
    mask = (
        working["feature_set"].astype(str).eq("core_with_lag")
        & working["successful_folds"].astype(int).eq(3)
        & finite_cv_metrics(working)
        & np.isfinite(pd.to_numeric(working["worst_fold_rmse"], errors="coerce"))
        & pd.to_numeric(working["worst_fold_rmse"], errors="coerce").le(MAX_STABLE_LOG_TARGET_WORST_RMSE)
    )
    return working.loc[mask].copy()


def shortlist_entry(row: pd.Series, application_track: str, selection_reason: str) -> dict[str, Any]:
    entry = row.to_dict()
    entry["application_track"] = application_track
    entry["time_cv_mean_mae"] = entry.get("mean_mae")
    entry["time_cv_std_mae"] = entry.get("std_mae")
    entry["time_cv_worst_fold_mae"] = entry.get("worst_fold_mae")
    entry["selection_reason"] = selection_reason
    return entry


def select_shortlist_entries(
    baseline_aggregate: pd.DataFrame,
    direct_aggregate: pd.DataFrame,
    residual_aggregate: pd.DataFrame,
    log_target_aggregate: pd.DataFrame,
    max_per_family: int = 2,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    baseline = baseline_aggregate.copy()
    direct = direct_aggregate.copy()
    residual = residual_aggregate.copy()
    log_target = log_target_aggregate.copy()

    forecast_baseline = baseline[baseline["model_run_id"].astype(str).eq("cv_baseline_lag_with_crop_median_fallback")]
    for _, row in forecast_baseline.iterrows():
        entries.append(shortlist_entry(row, FORECAST_TRACK, "forecast baseline using previous-year yield with crop median fallback"))

    forecast_direct = direct[direct["feature_set"].astype(str).eq("core_with_lag")]
    for _, row in rank_cv_results(forecast_direct).head(2).iterrows():
        entries.append(shortlist_entry(row, FORECAST_TRACK, "top direct time-CV model with core_with_lag"))

    for _, row in rank_cv_results(residual).head(2).iterrows():
        entries.append(shortlist_entry(row, FORECAST_TRACK, "top residual time-CV model correcting lag baseline"))

    stable_log = stable_log_target_rows(log_target)
    for _, row in rank_cv_results(stable_log).head(1).iterrows():
        entries.append(shortlist_entry(row, FORECAST_TRACK, "top stable log-target time-CV model with core_with_lag"))

    suitability_baseline = baseline[baseline["model_run_id"].astype(str).eq("cv_baseline_crop_median")]
    for _, row in suitability_baseline.iterrows():
        entries.append(shortlist_entry(row, SUITABILITY_TRACK, "suitability baseline without previous-year yield"))

    suitability_direct = direct[direct["feature_set"].astype(str).eq("core_without_lag")]
    for _, row in rank_cv_results(suitability_direct).head(2).iterrows():
        entries.append(shortlist_entry(row, SUITABILITY_TRACK, "top direct time-CV model ranked only within core_without_lag"))
    return entries


def grouped_shortlist_ids(entries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {
        FORECAST_TRACK: {
            "baseline_run_ids": [],
            "direct_run_ids": [],
            "residual_run_ids": [],
            "log_target_run_ids": [],
        },
        SUITABILITY_TRACK: {
            "baseline_run_ids": [],
            "direct_run_ids": [],
        },
    }
    for entry in entries:
        track = str(entry["application_track"])
        family = str(entry["experiment_family"])
        key = f"{family}_run_ids"
        if track in grouped and key in grouped[track]:
            grouped[track][key].append(str(entry["model_run_id"]))
    return grouped


def write_shortlist() -> list[dict[str, Any]]:
    baseline_aggregate = load_or_build_aggregate("baseline")
    direct_aggregate = load_or_build_aggregate("direct")
    residual_aggregate = load_or_build_aggregate("residual")
    log_target_aggregate = load_or_build_aggregate("log_target")
    entries = select_shortlist_entries(baseline_aggregate, direct_aggregate, residual_aggregate, log_target_aggregate)
    registry = all_config_registry()
    missing = [entry["model_run_id"] for entry in entries if entry["model_run_id"] not in registry]
    if missing:
        raise ValueError(f"Shortlist references unknown configs: {', '.join(missing)}")
    payload = {
        "selection_source": "time_cv_only",
        "selection_metric": "mean_mae",
        "tie_breaks": ["worst_fold_mae", "std_mae", "mean_rmse", "simpler_model"],
        "application_tracks": [FORECAST_TRACK, SUITABILITY_TRACK],
        **grouped_shortlist_ids(entries),
        "test_data_accessed": False,
        "created_at_utc": utc_now(),
        "entries": [sanitize(entry) for entry in entries],
    }
    write_json(SHORTLIST_PATH, payload)
    return entries


def load_shortlist() -> list[dict[str, Any]]:
    if not SHORTLIST_PATH.exists():
        return write_shortlist()
    with SHORTLIST_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_source") != "time_cv_only":
        raise ValueError("time_cv_shortlist.json must be selected only from time-CV results")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("Shortlist must not access test data")
    entries = list(payload.get("entries", []))
    if (
        FORECAST_TRACK not in payload
        or SUITABILITY_TRACK not in payload
        or any("application_track" not in entry for entry in entries)
    ):
        return write_shortlist()
    return entries


def run_validation_shortlist_sequence(
    entries: list[dict[str, Any]],
    train: pd.DataFrame,
    validation_frame: pd.DataFrame,
    manifest: dict[str, Any],
    resume: bool,
    force_run: str | None,
    random_state: int,
) -> list[dict[str, Any]]:
    registry = all_config_registry(random_state)
    results = []
    for entry in entries:
        model_run_id = str(entry["model_run_id"])
        application_track = str(entry.get("application_track", ""))
        if model_run_id not in registry:
            raise ValueError(f"Shortlisted run was not part of the configured CV search: {model_run_id}")
        config = registry[model_run_id]
        if application_track == SUITABILITY_TRACK:
            if config.experiment_family == "residual":
                raise ValueError("Suitability shortlist must not include residual models")
            if config.feature_set not in {"core_without_lag", "crop_only"}:
                raise ValueError("Suitability shortlist must not use lag features")
        existing_runs = read_runs()
        if should_skip_run(
            config,
            existing_runs,
            resume=resume,
            fold_id="validation_2011_2012",
            phase="validation-shortlist",
            force_run=force_run,
        ):
            print(f"skipped_completed={config.model_run_id} fold=validation_2011_2012")
            continue
        cv_metrics = {
            "mean_mae": entry.get("mean_mae", np.nan),
            "std_mae": entry.get("std_mae", np.nan),
            "worst_fold_mae": entry.get("worst_fold_mae", np.nan),
        }
        result = execute_validation_config(
            config,
            train,
            validation_frame,
            manifest,
            random_state=random_state,
            cv_metrics=cv_metrics,
            write_prediction_file=True,
        )
        result["application_track"] = application_track
        write_run_result(result)
        print(
            f"application_track={application_track} model_run_id={result['model_run_id']} status={result['status']} "
            f"validation_mae={result.get('mae', np.nan)} validation_rmse={result.get('rmse', np.nan)}"
        )
        results.append(result)
    return results


def validation_results() -> pd.DataFrame:
    rows = read_runs()
    results = rows[
        rows["phase"].astype(str).eq("validation-shortlist")
        & rows["status"].astype(str).eq("completed")
    ].copy()
    if results.empty:
        return pd.DataFrame()
    results = results.rename(
        columns={
            "mae": "validation_mae",
            "rmse": "validation_rmse",
            "r2": "validation_r2",
            "median_ae": "validation_median_ae",
        }
    )
    return results.sort_values(
        [
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "time_cv_mean_mae",
            "time_cv_std_mae",
            "model_run_id",
        ],
        ascending=[True, True, False, True, True, True],
    )


def write_tuned_validation_reports(validation_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    results = validation_results()
    if results.empty:
        raise ValueError("No completed validation-shortlist runs found")
    ensure_parent(TUNED_VALIDATION_RESULTS_PATH)
    results.to_csv(TUNED_VALIDATION_RESULTS_PATH, index=False)

    runtime = results[
        [
            "model_run_id",
            "application_track",
            "experiment_family",
            "model_name",
            "fit_rows",
            "evaluation_rows",
            "fit_seconds",
            "predict_seconds",
            "training_scope",
            "status",
            "warning_count",
            "warning_summary",
        ]
    ].copy()
    ensure_parent(TUNED_VALIDATION_RUNTIME_PATH)
    runtime.to_csv(TUNED_VALIDATION_RUNTIME_PATH, index=False)

    comparison = results[
        [
            "model_run_id",
            "application_track",
            "experiment_family",
            "model_name",
            "feature_set",
            "target_strategy",
            "time_cv_mean_mae",
            "time_cv_std_mae",
            "time_cv_worst_fold_mae",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "validation_median_ae",
            "test_data_accessed",
        ]
    ].copy()
    ensure_parent(TUNED_VALIDATION_COMPARISON_PATH)
    comparison.to_csv(TUNED_VALIDATION_COMPARISON_PATH, index=False)

    if validation_frame is not None:
        write_validation_prediction_sample(results)
        write_validation_subgroup_metrics(results)
    return results


def read_prediction_file(model_run_id: str) -> pd.DataFrame:
    path = prediction_path(model_run_id)
    if not path.exists():
        raise ValueError(f"Missing validation prediction file for {model_run_id}")
    return pd.read_csv(path)


def write_validation_prediction_sample(results: pd.DataFrame, max_rows: int = 1_000) -> pd.DataFrame:
    frames = []
    for model_run_id in results["model_run_id"].astype(str):
        predictions = read_prediction_file(model_run_id)
        predictions["model_run_id"] = model_run_id
        frames.append(predictions)
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sort_values(["canonical_crop_row_id", "model_run_id"]).head(max_rows)
    ensure_parent(TUNED_VALIDATION_PREDICTIONS_SAMPLE_PATH)
    sample.to_csv(TUNED_VALIDATION_PREDICTIONS_SAMPLE_PATH, index=False)
    return sample


def best_validation_rows(results: pd.DataFrame) -> dict[str, pd.Series]:
    if results.empty:
        raise ValueError("Validation results are required")
    ranked = results.copy()
    ranked["simplicity_rank"] = ranked["model_name"].map(MODEL_SIMPLICITY).fillna(999)
    ranked_all = ranked.sort_values(
        ["validation_mae", "validation_rmse", "validation_r2", "time_cv_mean_mae", "time_cv_std_mae", "simplicity_rank"],
        ascending=[True, True, False, True, True, True],
    )
    forecast = ranked[ranked["application_track"].astype(str).eq(FORECAST_TRACK)].copy()
    suitability = ranked[ranked["application_track"].astype(str).eq(SUITABILITY_TRACK)].copy()
    if forecast.empty:
        raise ValueError("Forecast validation results are required")
    if suitability.empty:
        raise ValueError("Suitability validation results are required")

    ranked_forecast = forecast.sort_values(
        ["validation_mae", "validation_rmse", "validation_r2", "time_cv_mean_mae", "time_cv_std_mae", "simplicity_rank"],
        ascending=[True, True, False, True, True, True],
    )
    forecast_trained = forecast[~forecast["experiment_family"].astype(str).eq("baseline")]
    ranked_forecast_trained = forecast_trained.sort_values(
        ["validation_mae", "validation_rmse", "validation_r2", "time_cv_mean_mae", "time_cv_std_mae", "simplicity_rank"],
        ascending=[True, True, False, True, True, True],
    )
    suitability_models = suitability[
        ~suitability["experiment_family"].astype(str).eq("baseline")
        & suitability["feature_set"].astype(str).eq("core_without_lag")
    ].copy()
    if suitability_models.empty:
        raise ValueError("Suitability validation needs a core_without_lag trained model")
    ranked_suitability = suitability_models.sort_values(
        ["validation_mae", "validation_rmse", "validation_r2", "time_cv_mean_mae", "time_cv_std_mae", "simplicity_rank"],
        ascending=[True, True, False, True, True, True],
    )
    output = {
        "best_overall": ranked_forecast.iloc[0],
        "best_trained": ranked_forecast_trained.iloc[0],
        "best_overall_forecast": ranked_forecast.iloc[0],
        "best_trained_forecast": ranked_forecast_trained.iloc[0],
        "best_suitability_model": ranked_suitability.iloc[0],
    }
    for family in ["direct", "residual", "log_target"]:
        family_rows = ranked[ranked["experiment_family"].astype(str).eq(family)]
        if not family_rows.empty:
            output[f"best_{family}"] = family_rows.sort_values(
                ["validation_mae", "validation_rmse", "validation_r2", "time_cv_mean_mae", "time_cv_std_mae", "simplicity_rank"],
                ascending=[True, True, False, True, True, True],
            ).iloc[0]
    return output


def write_validation_subgroup_metrics(results: pd.DataFrame) -> pd.DataFrame:
    winners = best_validation_rows(results)
    selected_ids = {
        str(winners["best_overall_forecast"]["model_run_id"]),
        str(winners["best_trained_forecast"]["model_run_id"]),
        str(winners["best_suitability_model"]["model_run_id"]),
    }
    subgroup_columns = ["Crop_canonical", "Season_canonical", "canonical_state_name", "Crop_Year"]
    all_rows = []
    for model_run_id in sorted(selected_ids):
        predictions = read_prediction_file(model_run_id)
        for column in subgroup_columns:
            grouped = predictions.groupby(column, dropna=False)
            for value, group in grouped:
                metrics = regression_metrics(group["target_yield"], group["prediction"])
                all_rows.append(
                    {
                        "model_run_id": model_run_id,
                        "subgroup_column": column,
                        "subgroup_value": value,
                        "row_count": int(len(group)),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"] if len(group) >= 2 else np.nan,
                        "reliable_group": int(len(group)) >= 20,
                    }
                )
    output = pd.DataFrame(all_rows)
    ensure_parent(TUNED_VALIDATION_SUBGROUP_METRICS_PATH)
    output.to_csv(TUNED_VALIDATION_SUBGROUP_METRICS_PATH, index=False)
    return output


def write_time_cv_all_results() -> pd.DataFrame:
    rows = read_runs()
    cv_rows = rows[rows["phase"].astype(str).str.startswith("cv-")].copy()
    ensure_parent(TIME_CV_ALL_RESULTS_PATH)
    cv_rows.to_csv(TIME_CV_ALL_RESULTS_PATH, index=False)
    return cv_rows


def load_time_cv_all_results() -> pd.DataFrame:
    if TIME_CV_ALL_RESULTS_PATH.exists():
        return pd.read_csv(TIME_CV_ALL_RESULTS_PATH)
    return write_time_cv_all_results()


def load_tuned_validation_results(default_results: pd.DataFrame | None = None) -> pd.DataFrame:
    if TUNED_VALIDATION_RESULTS_PATH.exists():
        return pd.read_csv(TUNED_VALIDATION_RESULTS_PATH)
    if default_results is not None:
        return default_results.copy()
    raise ValueError(f"Missing validation report: {TUNED_VALIDATION_RESULTS_PATH}")


def parse_hyperparameters(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected hyperparameters JSON object, found {type(parsed).__name__}")
    return parsed


def require_report_rows(frame: pd.DataFrame, run_id: str, source_name: str) -> pd.DataFrame:
    if "model_run_id" not in frame.columns:
        raise ValueError(f"{source_name} is missing model_run_id")
    rows = frame[frame["model_run_id"].astype(str).eq(run_id)].copy()
    if rows.empty:
        raise ValueError(f"{run_id} is missing from {source_name}")
    return rows


def validate_report_hyperparameters(
    run_id: str,
    rows: pd.DataFrame,
    source_name: str,
    expected: dict[str, Any],
) -> None:
    if "hyperparameters_json" not in rows.columns:
        raise ValueError(f"{source_name} is missing hyperparameters_json")
    actual_json = {
        json_dumps(parse_hyperparameters(value))
        for value in rows["hyperparameters_json"].dropna().tolist()
    }
    if not actual_json:
        actual_json = {json_dumps({})}
    if len(actual_json) != 1:
        raise ValueError(
            f"Hyperparameter mismatch for {run_id} in {source_name}: "
            f"expected {json_dumps(expected)}, found multiple values {sorted(actual_json)}"
        )
    actual = json.loads(next(iter(actual_json)))
    if actual != expected:
        raise ValueError(
            f"Hyperparameter mismatch for {run_id} in {source_name}: "
            f"expected {json_dumps(expected)}, found {json_dumps(actual)}"
        )


def validate_report_metadata(
    run_id: str,
    rows: pd.DataFrame,
    source_name: str,
    config: TuningConfig,
) -> None:
    expected_values = {
        "experiment_family": config.experiment_family,
        "model_name": config.model_name,
        "feature_set": config.feature_set,
        "target_strategy": config.target_strategy,
        "preprocessing_family": config.preprocessing_family,
    }
    for column, expected in expected_values.items():
        if column not in rows.columns:
            continue
        actual_values = {
            str(value)
            for value in rows[column].dropna().tolist()
            if str(value).strip()
        }
        if actual_values and actual_values != {str(expected)}:
            raise ValueError(
                f"Metadata mismatch for {run_id} in {source_name} column {column}: "
                f"expected {expected}, found {sorted(actual_values)}"
            )


def validate_frozen_track_reports(
    run_id: str,
    config: TuningConfig,
    cv_results: pd.DataFrame,
    validation_results: pd.DataFrame,
) -> None:
    for source_name, frame in [
        ("reports/time_cv_all_results.csv", cv_results),
        ("reports/tuned_validation_results.csv", validation_results),
    ]:
        rows = require_report_rows(frame, run_id, source_name)
        validate_report_metadata(run_id, rows, source_name, config)
        validate_report_hyperparameters(run_id, rows, source_name, config.hyperparameters)


def frozen_track_hyperparameters(track_name: str, config: TuningConfig, random_state: int) -> dict[str, Any]:
    hyperparameters = dict(config.hyperparameters)
    if track_name in {"suitability_model", "log_target_experiment"}:
        hyperparameters["random_state"] = random_state
    return hyperparameters


def build_frozen_tracks(
    winners: dict[str, pd.Series],
    registry: dict[str, TuningConfig],
    cv_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    actual_run_ids = {
        "forecast_baseline": str(winners["best_overall_forecast"]["model_run_id"]),
        "forecast_trained_model": str(winners["best_trained_forecast"]["model_run_id"]),
        "suitability_model": str(winners["best_suitability_model"]["model_run_id"]),
        "log_target_experiment": str(winners["best_log_target"]["model_run_id"]),
    }
    for track_name, expected_run_id in FROZEN_TRACK_RUN_IDS.items():
        actual_run_id = actual_run_ids.get(track_name)
        if actual_run_id != expected_run_id:
            raise ValueError(
                f"Frozen track {track_name} expected {expected_run_id}, "
                f"but selected run is {actual_run_id}"
            )
        if expected_run_id not in registry:
            raise ValueError(f"Missing registry config for frozen track {track_name}: {expected_run_id}")
        validate_frozen_track_reports(
            expected_run_id,
            registry[expected_run_id],
            cv_results,
            validation_results,
        )

    forecast_trained = registry[FROZEN_TRACK_RUN_IDS["forecast_trained_model"]]
    suitability = registry[FROZEN_TRACK_RUN_IDS["suitability_model"]]
    log_target = registry[FROZEN_TRACK_RUN_IDS["log_target_experiment"]]
    return {
        "forecast_baseline": {
            "run_id": FROZEN_TRACK_RUN_IDS["forecast_baseline"],
            "type": "baseline",
        },
        "forecast_trained_model": {
            "run_id": forecast_trained.model_run_id,
            "model": FROZEN_TRACK_MODEL_NAMES[forecast_trained.model_name],
            "feature_set": forecast_trained.feature_set,
            "target_strategy": forecast_trained.target_strategy,
            "preprocessing_family": forecast_trained.preprocessing_family,
            "hyperparameters": frozen_track_hyperparameters("forecast_trained_model", forecast_trained, random_state),
        },
        "suitability_model": {
            "run_id": suitability.model_run_id,
            "model": FROZEN_TRACK_MODEL_NAMES[suitability.model_name],
            "feature_set": suitability.feature_set,
            "target_strategy": suitability.target_strategy,
            "preprocessing_family": suitability.preprocessing_family,
            "hyperparameters": frozen_track_hyperparameters("suitability_model", suitability, random_state),
        },
        "log_target_experiment": {
            "run_id": log_target.model_run_id,
            "model": FROZEN_TRACK_MODEL_NAMES[log_target.model_name],
            "feature_set": log_target.feature_set,
            "target_strategy": "log1p",
            "preprocessing_family": log_target.preprocessing_family,
            "hyperparameters": frozen_track_hyperparameters("log_target_experiment", log_target, random_state),
        },
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(shown.columns)
    rows = shown.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_plots(results: pd.DataFrame) -> None:
    aggregates = [
        load_or_build_aggregate("baseline"),
        load_or_build_aggregate("direct"),
        load_or_build_aggregate("residual"),
        load_or_build_aggregate("log_target"),
    ]
    aggregate = pd.concat([frame for frame in aggregates if not frame.empty], ignore_index=True)
    if not aggregate.empty:
        plot = aggregate.sort_values("mean_mae")
        ensure_parent(TIME_CV_MAE_PLOT_PATH)
        plt.figure(figsize=(12, max(4, len(plot) * 0.25)))
        plt.barh(plot["model_run_id"], plot["mean_mae"])
        plt.xlabel("Mean CV MAE")
        plt.ylabel("Configuration")
        plt.tight_layout()
        plt.savefig(TIME_CV_MAE_PLOT_PATH)
        plt.close()

        stability = aggregate.sort_values("std_mae")
        ensure_parent(TIME_CV_STABILITY_PLOT_PATH)
        plt.figure(figsize=(12, max(4, len(stability) * 0.25)))
        plt.barh(stability["model_run_id"], stability["std_mae"])
        plt.xlabel("CV MAE standard deviation")
        plt.ylabel("Configuration")
        plt.tight_layout()
        plt.savefig(TIME_CV_STABILITY_PLOT_PATH)
        plt.close()

    if not results.empty:
        plot = results.sort_values("validation_mae")
        ensure_parent(TUNED_VALIDATION_MAE_PLOT_PATH)
        plt.figure(figsize=(10, max(4, len(plot) * 0.35)))
        plt.barh(plot["model_run_id"], plot["validation_mae"])
        plt.xlabel("Validation MAE")
        plt.ylabel("Shortlisted configuration")
        plt.tight_layout()
        plt.savefig(TUNED_VALIDATION_MAE_PLOT_PATH)
        plt.close()


def build_frozen_tuned_configuration(
    results: pd.DataFrame,
    cv_results: pd.DataFrame | None = None,
    validation_report: pd.DataFrame | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    report_results = validation_report.copy() if validation_report is not None else load_tuned_validation_results(results)
    cv_report_results = cv_results.copy() if cv_results is not None else load_time_cv_all_results()
    winners = best_validation_rows(report_results)
    selected = winners["best_trained_forecast"]
    best_overall_forecast = winners["best_overall_forecast"]
    best_suitability = winners["best_suitability_model"]
    registry = all_config_registry(random_state)
    selected_config = registry[str(selected["model_run_id"])]
    frozen_tracks = build_frozen_tracks(
        winners,
        registry,
        cv_report_results,
        report_results,
        random_state,
    )
    payload = {
        "selected_run_id": selected_config.model_run_id,
        "selected_experiment_family": selected_config.experiment_family,
        "selected_model_name": selected_config.model_name,
        "selected_feature_set": selected_config.feature_set,
        "selected_target_strategy": selected_config.target_strategy,
        "selected_hyperparameters": selected_config.hyperparameters,
        "selected_preprocessing_family": selected_config.preprocessing_family,
        "time_cv_folds": [fold.__dict__ for fold in get_time_cv_folds()],
        "time_cv_metrics": {
            "mean_mae": selected.get("time_cv_mean_mae"),
            "std_mae": selected.get("time_cv_std_mae"),
            "worst_fold_mae": selected.get("time_cv_worst_fold_mae"),
        },
        "validation_metrics": {
            "mae": selected.get("validation_mae"),
            "rmse": selected.get("validation_rmse"),
            "r2": selected.get("validation_r2"),
            "median_ae": selected.get("validation_median_ae"),
        },
        "best_overall_validation_run_id": best_overall_forecast.get("model_run_id"),
        "best_trained_model_validation_run_id": selected.get("model_run_id"),
        "best_overall_forecast_validation_run": best_overall_forecast.get("model_run_id"),
        "best_trained_forecast_validation_model": selected.get("model_run_id"),
        "best_suitability_validation_model": best_suitability.get("model_run_id"),
        "best_direct_model": winners.get("best_direct", pd.Series(dtype=object)).get("model_run_id"),
        "best_residual_model": winners.get("best_residual", pd.Series(dtype=object)).get("model_run_id"),
        "best_log_target_model": winners.get("best_log_target", pd.Series(dtype=object)).get("model_run_id"),
        "train_period": "1997-2010",
        "validation_period": "2011-2012",
        "test_period": "2013-2014",
        "test_data_accessed": False,
        "test_used_for_selection": False,
        "frozen_tracks": frozen_tracks,
        "created_at_utc": utc_now(),
    }
    write_json(FROZEN_TUNED_CONFIG_PATH, payload)
    return payload


def read_original_benchmark_summary() -> tuple[str, str]:
    baseline = "not available"
    model = "not available"
    if validation.BASELINE_REPORT_PATH.exists():
        frame = pd.read_csv(validation.BASELINE_REPORT_PATH)
        completed = frame[frame["status"].astype(str).eq("completed")]
        if not completed.empty:
            row = completed.sort_values("mae").iloc[0]
            baseline = f"{row['model_run_id']} MAE={row['mae']} RMSE={row['rmse']}"
    if validation.MODEL_RESULTS_PATH.exists():
        frame = pd.read_csv(validation.MODEL_RESULTS_PATH)
        completed = frame[
            frame["status"].astype(str).eq("completed")
            & frame["phase"].astype(str).eq("models")
        ]
        if not completed.empty:
            row = completed.sort_values("mae").iloc[0]
            model = f"{row['model_run_id']} MAE={row['mae']} RMSE={row['rmse']}"
    return baseline, model


def write_summary(results: pd.DataFrame, frozen_config: dict[str, Any]) -> None:
    baseline_cv = load_or_build_aggregate("baseline")
    direct_cv = load_or_build_aggregate("direct")
    residual_cv = load_or_build_aggregate("residual")
    log_cv = load_or_build_aggregate("log_target")
    shortlist_entries = pd.DataFrame(load_shortlist())
    original_baseline, original_model = read_original_benchmark_summary()
    residual_best = rank_cv_results(residual_cv).head(1)
    baseline_best = rank_cv_results(baseline_cv).head(1)
    residual_beat_lag = False
    if not residual_best.empty and not baseline_best.empty:
        lag_row = baseline_cv[baseline_cv["model_run_id"].eq("cv_baseline_lag_with_crop_median_fallback")]
        if not lag_row.empty:
            residual_beat_lag = bool(float(residual_best.iloc[0]["mean_mae"]) < float(lag_row.iloc[0]["mean_mae"]))
    log_best = rank_cv_results(log_cv).head(1)
    direct_best = rank_cv_results(direct_cv).head(1)
    log_helped = False
    if not log_best.empty and not direct_best.empty:
        log_helped = bool(float(log_best.iloc[0]["mean_mae"]) < float(direct_best.iloc[0]["mean_mae"]))

    lines = [
        "# Time-Aware Model Tuning Summary",
        "",
        "## Time-CV folds",
        markdown_table(pd.DataFrame([fold.__dict__ for fold in get_time_cv_folds()]), list(TimeFold.__dataclass_fields__.keys())),
        "",
        "## Rolling lag assumption",
        ROLLING_LAG_ASSUMPTION,
        "",
        "## Baseline CV results",
        markdown_table(baseline_cv, ["model_run_id", "mean_mae", "std_mae", "mean_rmse", "worst_fold_mae"]),
        "",
        "## Direct model CV results",
        markdown_table(direct_cv, ["model_run_id", "model_name", "feature_set", "mean_mae", "std_mae", "worst_fold_mae"], 12),
        "",
        "## Residual model CV results",
        markdown_table(residual_cv, ["model_run_id", "model_name", "mean_mae", "std_mae", "worst_fold_mae"], 12),
        "",
        "## Log-target model CV results",
        markdown_table(log_cv, ["model_run_id", "model_name", "mean_mae", "std_mae", "worst_fold_mae"], 12),
        "",
        "## Selected time-CV shortlist",
        markdown_table(
            shortlist_entries,
            [
                "application_track",
                "model_run_id",
                "experiment_family",
                "feature_set",
                "time_cv_mean_mae",
                "time_cv_worst_fold_mae",
                "selection_reason",
            ],
            20,
        ),
        "",
        "## Validation shortlist results",
        markdown_table(
            results,
            [
                "application_track",
                "model_run_id",
                "experiment_family",
                "model_name",
                "feature_set",
                "validation_mae",
                "validation_rmse",
                "validation_r2",
                "time_cv_mean_mae",
            ],
            20,
        ),
        "",
        "## Original validation benchmark comparison",
        f"- Original best baseline: {original_baseline}",
        f"- Original best trained model: {original_model}",
        "",
        "## Selected frozen configuration",
        f"- Best forecast validation result overall: {frozen_config['best_overall_forecast_validation_run']}",
        f"- Best trained forecast validation model: {frozen_config['best_trained_forecast_validation_model']}",
        f"- Best suitability validation model: {frozen_config['best_suitability_validation_model']}",
        f"- Selected run: {frozen_config['selected_run_id']}",
        f"- Residual beat lag baseline in mean CV MAE: {str(residual_beat_lag).lower()}",
        f"- Log-target improved over direct in mean CV MAE: {str(log_helped).lower()}",
        "",
        "## Test usage",
        "- Test 2013-2014 was not opened.",
        "- test_data_accessed: false",
        "- test_used_for_selection: false",
    ]
    ensure_parent(SUMMARY_PATH)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_outputs(random_state: int = 42) -> dict[str, Any]:
    write_baseline_fold_report()
    write_cv_aggregate_report("direct")
    write_cv_aggregate_report("residual")
    write_cv_aggregate_report("log_target")
    write_time_cv_all_results()
    entries = load_shortlist()
    if not entries:
        raise ValueError("Shortlist is empty")
    validation_frame = safe_read_parquet(VALIDATION_PATH)
    results = write_tuned_validation_reports(validation_frame)
    frozen_config = build_frozen_tuned_configuration(results, random_state=random_state)
    write_plots(results)
    write_summary(results, frozen_config)
    return frozen_config


def run_cv_phase(
    experiment_family: str,
    configs: list[TuningConfig],
    resume: bool,
    force_run: str | None,
    random_state: int,
) -> None:
    folds = get_time_cv_folds()
    validate_time_cv_folds(folds)
    train, manifest = load_train_only()
    run_config_sequence(configs, folds, train, manifest, resume, force_run, random_state)
    if experiment_family == "baseline":
        write_baseline_fold_report()
        aggregate = aggregate_cv_results("baseline")
    else:
        aggregate = write_cv_aggregate_report(experiment_family)
    if not aggregate.empty:
        best = rank_cv_results(aggregate).iloc[0]
        print(
            f"best_{experiment_family}_cv_run={best['model_run_id']} "
            f"mean_mae={best['mean_mae']} worst_fold_mae={best['worst_fold_mae']}"
        )


def run_validation_shortlist_phase(resume: bool, force_run: str | None, random_state: int) -> None:
    train, validation_frame, manifest = load_train_validation()
    entries = load_shortlist()
    run_validation_shortlist_sequence(entries, train, validation_frame, manifest, resume, force_run, random_state)
    results = write_tuned_validation_reports(validation_frame)
    winners = best_validation_rows(results)
    print(f"best_overall_forecast_validation_run={winners['best_overall_forecast']['model_run_id']}")
    print(f"best_trained_forecast_validation_model={winners['best_trained_forecast']['model_run_id']}")
    print(f"best_suitability_validation_model={winners['best_suitability_model']['model_run_id']}")


def run_phase(phase: str, resume: bool, force_run: str | None, random_state: int) -> None:
    if phase == "cv-baselines":
        run_cv_phase("baseline", baseline_configs(), resume, force_run, random_state)
        return
    if phase == "cv-direct":
        run_cv_phase("direct", direct_configs(random_state), resume, force_run, random_state)
        return
    if phase == "cv-residual":
        run_cv_phase("residual", residual_configs(random_state), resume, force_run, random_state)
        return
    if phase == "cv-log-target":
        run_cv_phase("log_target", log_target_configs(random_state), resume, force_run, random_state)
        write_shortlist()
        return
    if phase == "validation-shortlist":
        run_validation_shortlist_phase(resume, force_run, random_state)
        return
    if phase == "finalize":
        frozen_config = finalize_outputs(random_state)
        print(f"selected_run_id={frozen_config['selected_run_id']}")
        print(f"best_overall_validation_run_id={frozen_config['best_overall_validation_run_id']}")
        print(f"test_data_accessed={str(frozen_config['test_data_accessed']).lower()}")
        return
    raise ValueError(f"Unsupported phase: {phase}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-aware tuning without opening the final test split.")
    parser.add_argument(
        "--phase",
        choices=[
            "cv-baselines",
            "cv-direct",
            "cv-residual",
            "cv-log-target",
            "validation-shortlist",
            "finalize",
        ],
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-run", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_phase(args.phase, args.resume, args.force_run, args.random_state)
    except Exception as error:
        print(f"Time-aware tuning failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
