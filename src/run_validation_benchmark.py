from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_local_venv_site_packages() -> None:
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        REPO_ROOT / ".venv" / "Lib" / "site-packages",
        REPO_ROOT / ".venv" / "lib" / version_dir / "site-packages",
        REPO_ROOT.parent / ".venv" / "Lib" / "site-packages",
        REPO_ROOT.parent / ".venv" / "lib" / version_dir / "site-packages",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


add_local_venv_site_packages()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor


TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_1997_2010.parquet"
VALIDATION_PATH = REPO_ROOT / "data" / "processed" / "validation_2011_2012.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_2013_2014.parquet"
FEATURE_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "model_feature_manifest.json"
SELECTED_FEATURE_SET_PATH = REPO_ROOT / "data" / "reference" / "selected_validation_feature_set.json"
FROZEN_MODEL_CONFIG_PATH = REPO_ROOT / "data" / "reference" / "frozen_model_configuration.json"

RUNS_PATH = REPO_ROOT / "data" / "interim" / "validation_benchmark_runs.csv"
PREDICTIONS_DIR = REPO_ROOT / "data" / "interim" / "validation_predictions"
MODELS_DIR = REPO_ROOT / "data" / "interim" / "validation_models"

BASELINE_REPORT_PATH = REPO_ROOT / "reports" / "validation_baseline_results.csv"
FEATURE_SET_REPORT_PATH = REPO_ROOT / "reports" / "validation_feature_set_comparison.csv"
MODEL_RESULTS_PATH = REPO_ROOT / "reports" / "validation_model_results.csv"
RUNTIME_RESULTS_PATH = REPO_ROOT / "reports" / "validation_runtime_results.csv"
SUBGROUP_METRICS_PATH = REPO_ROOT / "reports" / "validation_subgroup_metrics.csv"
PREDICTIONS_SAMPLE_PATH = REPO_ROOT / "reports" / "validation_predictions_sample.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "validation_benchmark_summary.md"
MAE_PLOT_PATH = REPO_ROOT / "reports" / "validation_mae.png"
RMSE_PLOT_PATH = REPO_ROOT / "reports" / "validation_rmse.png"
R2_PLOT_PATH = REPO_ROOT / "reports" / "validation_r2.png"

TARGET_COLUMN = "target_yield"
EXPECTED_TRAIN_ROWS = 202_166
EXPECTED_VALIDATION_ROWS = 32_388
TRAIN_YEARS = set(range(1997, 2011))
VALIDATION_YEARS = {2011, 2012}
TEST_DATA_ACCESSED = False
LAG_INTERPRETATION = (
    "Lag features assume that the previous year's official crop yield is available at prediction time."
)
FORBIDDEN_TEST_USAGE = (
    "The 2013-2014 test set was not used for preprocessing, feature selection, "
    "hyperparameter tuning or model selection."
)

RUN_COLUMNS = [
    "model_run_id",
    "phase",
    "model_name",
    "model_family",
    "feature_set",
    "preprocessing_family",
    "hyperparameters_json",
    "model_parameters",
    "training_scope",
    "train_rows_available",
    "train_rows_used",
    "validation_rows",
    "fit_seconds",
    "predict_seconds",
    "mae",
    "rmse",
    "r2",
    "median_ae",
    "lag_subset_mae",
    "lag_subset_rows",
    "status",
    "warning_count",
    "warning_summary",
    "error_type",
    "error_message",
    "random_state",
    "test_data_accessed",
]

SIMPLE_MODEL_ORDER = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "DecisionTree",
    "RandomForest",
    "LinearSVR",
    "KNN",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    model_run_id: str
    phase: str
    model_id: str
    model_name: str
    model_family: str
    feature_set: str
    preprocessing_family: str
    hyperparameters: dict[str, Any]
    training_scope: str = "full_train"


class CropMedianRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, crop_column: str = "Crop_canonical") -> None:
        self.crop_column = crop_column

    def fit(self, x: pd.DataFrame, y: Iterable[float]) -> "CropMedianRegressor":
        if self.crop_column not in x.columns:
            raise ValueError(f"Missing crop column: {self.crop_column}")
        target = pd.Series(y, index=x.index, dtype="float64")
        self.global_median_ = float(target.median())
        self.crop_medians_ = target.groupby(x[self.crop_column]).median().to_dict()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.crop_column not in x.columns:
            raise ValueError(f"Missing crop column: {self.crop_column}")
        predictions = x[self.crop_column].map(self.crop_medians_).fillna(self.global_median_)
        return predictions.astype("float64").to_numpy()


class LagWithCropMedianFallbackRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        crop_column: str = "Crop_canonical",
        lag_available_column: str = "lag_available",
        lag_yield_column: str = "lag_yield_1y",
    ) -> None:
        self.crop_column = crop_column
        self.lag_available_column = lag_available_column
        self.lag_yield_column = lag_yield_column

    def fit(self, x: pd.DataFrame, y: Iterable[float]) -> "LagWithCropMedianFallbackRegressor":
        self.crop_baseline_ = CropMedianRegressor(crop_column=self.crop_column).fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        required = {self.crop_column, self.lag_available_column, self.lag_yield_column}
        missing = sorted(required - set(x.columns))
        if missing:
            raise ValueError(f"Missing columns for lag baseline: {', '.join(missing)}")
        fallback = pd.Series(self.crop_baseline_.predict(x), index=x.index)
        lag_values = pd.to_numeric(x[self.lag_yield_column], errors="coerce")
        use_lag = x[self.lag_available_column].astype(int).eq(1) & lag_values.notna()
        fallback.loc[use_lag] = lag_values.loc[use_lag]
        return fallback.astype("float64").to_numpy()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(sanitize_for_json(value), indent=2, sort_keys=False, allow_nan=False),
        encoding="utf-8",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def guard_not_test_path(path: Path) -> None:
    if path.resolve() == TEST_PATH.resolve():
        raise ValueError("The 2013-2014 test dataset must not be loaded by this benchmark")


def safe_read_parquet(path: Path) -> pd.DataFrame:
    guard_not_test_path(path)
    return pd.read_parquet(path)


def load_manifest(path: Path = FEATURE_MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if TARGET_COLUMN != manifest.get("target_column"):
        raise ValueError("Feature manifest target_column does not match target_yield")
    return manifest


def load_benchmark_inputs(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    manifest_path: Path = FEATURE_MANIFEST_PATH,
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS,
    expected_validation_rows: int | None = EXPECTED_VALIDATION_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = safe_read_parquet(train_path)
    validation = safe_read_parquet(validation_path)
    manifest = load_manifest(manifest_path)
    validate_benchmark_inputs(train, validation, manifest, expected_train_rows, expected_validation_rows)
    return train, validation, manifest


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def validate_target(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, [TARGET_COLUMN], label)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or np.isinf(target.to_numpy()).any():
        raise ValueError(f"{label} target_yield contains NaN or infinity")


def validate_years(frame: pd.DataFrame, expected_years: set[int], label: str) -> None:
    require_columns(frame, ["Crop_Year"], label)
    observed_years = set(frame["Crop_Year"].astype(int).unique())
    if observed_years - expected_years:
        raise ValueError(f"{label} contains years outside expected period: {sorted(observed_years)}")


def validate_feature_set(feature_set_name: str, features: list[str], frame_columns: Iterable[str], manifest: dict[str, Any]) -> None:
    forbidden = set(manifest.get("forbidden_leakage_columns", []))
    if TARGET_COLUMN in features:
        raise ValueError(f"{feature_set_name} includes target_yield")
    forbidden_used = sorted(forbidden.intersection(features))
    if forbidden_used:
        raise ValueError(f"{feature_set_name} contains forbidden leakage columns: {', '.join(forbidden_used)}")
    missing = sorted(set(features) - set(frame_columns))
    if missing:
        raise ValueError(f"{feature_set_name} is missing approved features: {', '.join(missing)}")


def validate_benchmark_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict[str, Any],
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS,
    expected_validation_rows: int | None = EXPECTED_VALIDATION_ROWS,
) -> None:
    if expected_train_rows is not None and len(train) != expected_train_rows:
        raise ValueError(f"Expected {expected_train_rows} train rows, found {len(train)}")
    if expected_validation_rows is not None and len(validation) != expected_validation_rows:
        raise ValueError(f"Expected {expected_validation_rows} validation rows, found {len(validation)}")
    validate_years(train, TRAIN_YEARS, "train")
    validate_years(validation, VALIDATION_YEARS, "validation")
    validate_target(train, "train")
    validate_target(validation, "validation")
    for name, features in manifest.get("feature_sets", {}).items():
        validate_feature_set(name, list(features), train.columns, manifest)
        validate_feature_set(name, list(features), validation.columns, manifest)


def categorical_columns_for_features(features: list[str], manifest: dict[str, Any]) -> list[str]:
    categorical = set(manifest.get("categorical_features", []))
    return [column for column in features if column in categorical]


def numeric_columns_for_features(features: list[str], manifest: dict[str, Any]) -> list[str]:
    categorical = set(categorical_columns_for_features(features, manifest))
    return [column for column in features if column not in categorical]


def make_one_hot_encoder() -> OneHotEncoder:
    params: dict[str, Any] = {
        "handle_unknown": "ignore",
        "min_frequency": 5,
    }
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = True
    else:
        params["sparse"] = True
    return OneHotEncoder(**params)


def make_linear_preprocessor(features: list[str], manifest: dict[str, Any]) -> ColumnTransformer:
    categorical = categorical_columns_for_features(features, manifest)
    numeric = numeric_columns_for_features(features, manifest)
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def make_tree_preprocessor(features: list[str], manifest: dict[str, Any]) -> ColumnTransformer:
    categorical = categorical_columns_for_features(features, manifest)
    numeric = numeric_columns_for_features(features, manifest)
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        sparse_threshold=0.0,
    )


def make_knn_preprocessor(features: list[str], manifest: dict[str, Any]) -> ColumnTransformer:
    categorical = categorical_columns_for_features(features, manifest)
    numeric = numeric_columns_for_features(features, manifest)
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        sparse_threshold=0.0,
    )


def sklearn_version_tuple() -> tuple[int, int]:
    pieces = sklearn_version.split(".")[:2]
    try:
        return int(pieces[0]), int(pieces[1])
    except (IndexError, ValueError):
        return 0, 0


def make_linearsvr(c_value: float, random_state: int) -> LinearSVR:
    params: dict[str, Any] = {
        "C": c_value,
        "epsilon": 0.0,
        "max_iter": 10_000,
        "random_state": random_state,
    }
    if sklearn_version_tuple() >= (1, 3):
        params["dual"] = "auto"
    return LinearSVR(**params)


def make_linear_regression() -> LinearRegression:
    params: dict[str, Any] = {"fit_intercept": True}
    if "n_jobs" in inspect.signature(LinearRegression).parameters:
        params["n_jobs"] = -1
    return LinearRegression(**params)


def make_pipeline_for_config(config: BenchmarkConfig, features: list[str], manifest: dict[str, Any], random_state: int) -> Any:
    if config.preprocessing_family == "linear":
        preprocessor = make_linear_preprocessor(features, manifest)
    elif config.preprocessing_family == "tree":
        preprocessor = make_tree_preprocessor(features, manifest)
    elif config.preprocessing_family == "knn":
        preprocessor = make_knn_preprocessor(features, manifest)
    else:
        raise ValueError(f"Unsupported preprocessing family: {config.preprocessing_family}")

    if config.model_name == "LinearRegression":
        model = make_linear_regression()
    elif config.model_name == "Ridge":
        model = Ridge(**config.hyperparameters)
    elif config.model_name == "Lasso":
        model = Lasso(**config.hyperparameters)
    elif config.model_name == "DecisionTree":
        params = dict(config.hyperparameters)
        params["random_state"] = random_state
        model = DecisionTreeRegressor(**params)
    elif config.model_name == "RandomForest":
        params = dict(config.hyperparameters)
        params["random_state"] = random_state
        model = RandomForestRegressor(**params)
    elif config.model_name == "LinearSVR":
        model = make_linearsvr(float(config.hyperparameters["C"]), random_state)
    elif config.model_name == "KNN":
        model = KNeighborsRegressor(**config.hyperparameters)
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype="float64")
    pred = np.asarray(list(y_pred), dtype="float64")
    errors = true - pred
    abs_errors = np.abs(errors)
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(math.sqrt(np.mean(np.square(errors)))),
        "r2": float(r2_score(true, pred)),
        "median_ae": float(np.median(abs_errors)),
    }


def lag_subset_mae(validation: pd.DataFrame, predictions: Iterable[float]) -> tuple[float | None, int]:
    if "lag_available" not in validation.columns:
        return None, 0
    mask = validation["lag_available"].astype(int).eq(1)
    rows = int(mask.sum())
    if rows == 0:
        return None, 0
    pred = np.asarray(list(predictions), dtype="float64")
    subset_metrics = compute_regression_metrics(validation.loc[mask, TARGET_COLUMN], pred[mask.to_numpy()])
    return subset_metrics["mae"], rows


def empty_result(config: BenchmarkConfig, random_state: int, train_rows_available: int, validation_rows: int) -> dict[str, Any]:
    parameters = json_dumps(config.hyperparameters)
    return {
        "model_run_id": config.model_run_id,
        "phase": config.phase,
        "model_name": config.model_name,
        "model_family": config.model_family,
        "feature_set": config.feature_set,
        "preprocessing_family": config.preprocessing_family,
        "hyperparameters_json": parameters,
        "model_parameters": parameters,
        "training_scope": config.training_scope,
        "train_rows_available": train_rows_available,
        "train_rows_used": 0,
        "validation_rows": validation_rows,
        "fit_seconds": np.nan,
        "predict_seconds": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "median_ae": np.nan,
        "lag_subset_mae": np.nan,
        "lag_subset_rows": 0,
        "status": "failed",
        "warning_count": 0,
        "warning_summary": "",
        "error_type": "",
        "error_message": "",
        "random_state": random_state,
        "test_data_accessed": False,
    }


def prediction_path(model_run_id: str) -> Path:
    return PREDICTIONS_DIR / f"{model_run_id}.csv"


def write_predictions(model_run_id: str, validation: pd.DataFrame, predictions: Iterable[float]) -> None:
    ensure_parent(prediction_path(model_run_id))
    frame = pd.DataFrame(
        {
            "canonical_crop_row_id": validation["canonical_crop_row_id"].astype(str),
            "prediction": np.asarray(list(predictions), dtype="float64"),
        }
    )
    frame.to_csv(prediction_path(model_run_id), index=False)


def sample_knn_train(train: pd.DataFrame, max_rows: int = 15_000, random_state: int = 42) -> pd.DataFrame:
    if len(train) <= max_rows:
        return train.copy()
    require_columns(train, ["Crop_Year", "Crop_canonical"], "KNN train sample")
    rng = np.random.default_rng(random_state)
    working = train.copy()
    working["_stratum"] = working["Crop_Year"].astype(str) + "|" + working["Crop_canonical"].astype(str)
    group_sizes = working["_stratum"].value_counts().sort_index()
    ideal = group_sizes * (max_rows / len(working))
    counts = np.floor(ideal).astype(int)
    counts[counts < 1] = 1
    if counts.sum() > max_rows:
        priority = ideal.sort_values(ascending=False).index.tolist()
        keep = set(priority[:max_rows])
        counts = pd.Series(0, index=counts.index, dtype=int)
        counts.loc[list(keep)] = 1
    else:
        remainders = (ideal - np.floor(ideal)).sort_values(ascending=False)
        remaining = max_rows - int(counts.sum())
        for stratum in remainders.index[:remaining]:
            counts.loc[stratum] += 1

    sampled_parts = []
    for stratum, count in counts[counts > 0].items():
        group = working[working["_stratum"].eq(stratum)]
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sampled_parts.append(group.sample(n=min(int(count), len(group)), random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=False).drop(columns=["_stratum"])
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.sort_index()


def run_baseline_estimator(config: BenchmarkConfig, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    if config.model_run_id == "baseline_global_median":
        model = DummyRegressor(strategy="median")
        return model, train[[]], validation[[]]
    if config.model_run_id == "baseline_crop_median":
        model = CropMedianRegressor()
        return model, train[["Crop_canonical"]], validation[["Crop_canonical"]]
    if config.model_run_id == "baseline_lag_with_crop_median_fallback":
        columns = ["Crop_canonical", "lag_available", "lag_yield_1y"]
        model = LagWithCropMedianFallbackRegressor()
        return model, train[columns], validation[columns]
    raise ValueError(f"Unsupported baseline: {config.model_run_id}")


def execute_model_config(
    config: BenchmarkConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict[str, Any],
    random_state: int = 42,
    write_prediction_file: bool = True,
) -> dict[str, Any]:
    result = empty_result(config, random_state, len(train), len(validation))
    try:
        if config.phase == "baselines":
            estimator, x_train, x_validation = run_baseline_estimator(config, train, validation)
            train_rows_used = len(train)
        else:
            features = list(manifest["feature_sets"][config.feature_set])
            validate_feature_set(config.feature_set, features, train.columns, manifest)
            validate_feature_set(config.feature_set, features, validation.columns, manifest)
            estimator = make_pipeline_for_config(config, features, manifest, random_state)
            if config.model_name == "KNN":
                sampled_train = sample_knn_train(train, max_rows=15_000, random_state=random_state)
                train_rows_used = len(sampled_train)
                x_train = sampled_train[features]
                y_train = sampled_train[TARGET_COLUMN]
            else:
                train_rows_used = len(train)
                x_train = train[features]
                y_train = train[TARGET_COLUMN]
            x_validation = validation[features]
        if config.phase == "baselines":
            y_train = train[TARGET_COLUMN]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_start = time.perf_counter()
            estimator.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            predictions = estimator.predict(x_validation)
            predict_seconds = time.perf_counter() - predict_start

        metrics = compute_regression_metrics(validation[TARGET_COLUMN], predictions)
        subset_mae, subset_rows = lag_subset_mae(validation, predictions)
        warning_messages = sorted({str(item.message) for item in caught})

        result.update(metrics)
        result.update(
            {
                "train_rows_used": train_rows_used,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "lag_subset_mae": subset_mae,
                "lag_subset_rows": subset_rows,
                "status": "completed",
                "warning_count": len(caught),
                "warning_summary": " | ".join(message[:250] for message in warning_messages),
                "test_data_accessed": False,
            }
        )
        if write_prediction_file:
            write_predictions(config.model_run_id, validation, predictions)
    except MemoryError as error:
        result.update({"status": "failed", "error_type": "MemoryError", "error_message": str(error)})
    except Exception as error:
        result.update({"status": "failed", "error_type": type(error).__name__, "error_message": str(error)})
    return result


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
    existing = existing[~existing["model_run_id"].astype(str).eq(str(result["model_run_id"]))]
    new_row = pd.DataFrame([{column: result.get(column, np.nan) for column in RUN_COLUMNS}])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(path, index=False)


def has_completed_run(existing_runs: pd.DataFrame, model_run_id: str) -> bool:
    if existing_runs.empty:
        return False
    matches = existing_runs[
        existing_runs["model_run_id"].astype(str).eq(model_run_id)
        & existing_runs["status"].astype(str).eq("completed")
    ]
    return not matches.empty


def should_skip_run(
    config: BenchmarkConfig,
    existing_runs: pd.DataFrame,
    resume: bool,
    force_model: str | None = None,
) -> bool:
    if force_model is not None and force_model not in {config.model_run_id, config.model_id}:
        return True
    if force_model is not None:
        return False
    return resume and has_completed_run(existing_runs, config.model_run_id)


def run_config_sequence(
    configs: list[BenchmarkConfig],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: dict[str, Any],
    resume: bool,
    force_model: str | None,
    random_state: int,
    evaluator: Callable[[BenchmarkConfig, pd.DataFrame, pd.DataFrame, dict[str, Any], int], dict[str, Any]] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    evaluator = evaluator or (
        lambda config, train_frame, validation_frame, manifest_dict, seed: execute_model_config(
            config,
            train_frame,
            validation_frame,
            manifest_dict,
            random_state=seed,
        )
    )
    writer = writer or write_run_result
    results = []
    for config in configs:
        existing_runs = read_runs()
        if should_skip_run(config, existing_runs, resume=resume, force_model=force_model):
            print(f"skipped_completed={config.model_run_id}")
            continue
        try:
            result = evaluator(config, train, validation, manifest, random_state)
        except MemoryError as error:
            result = empty_result(config, random_state, len(train), len(validation))
            result.update({"error_type": "MemoryError", "error_message": str(error)})
        except Exception as error:
            result = empty_result(config, random_state, len(train), len(validation))
            result.update({"error_type": type(error).__name__, "error_message": str(error)})
        writer(result)
        print(
            f"model_run_id={result['model_run_id']} status={result['status']} "
            f"mae={result.get('mae', np.nan)} rmse={result.get('rmse', np.nan)}"
        )
        results.append(result)
    return results


def baseline_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            model_run_id="baseline_global_median",
            phase="baselines",
            model_id="baseline_global_median",
            model_name="GlobalMedian",
            model_family="baseline",
            feature_set="none",
            preprocessing_family="none",
            hyperparameters={"strategy": "median"},
        ),
        BenchmarkConfig(
            model_run_id="baseline_crop_median",
            phase="baselines",
            model_id="baseline_crop_median",
            model_name="CropMedian",
            model_family="baseline",
            feature_set="crop_only",
            preprocessing_family="none",
            hyperparameters={"fallback": "global_train_median"},
        ),
        BenchmarkConfig(
            model_run_id="baseline_lag_with_crop_median_fallback",
            phase="baselines",
            model_id="baseline_lag_with_crop_median_fallback",
            model_name="LagWithCropMedianFallback",
            model_family="baseline",
            feature_set="lag_with_crop_fallback",
            preprocessing_family="none",
            hyperparameters={"fallback": "crop_train_median"},
        ),
    ]


def feature_set_configs(random_state: int = 42) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for feature_set in ["core_without_lag", "core_with_lag"]:
        configs.append(
            BenchmarkConfig(
                model_run_id=f"feature_set_ridge_{feature_set}",
                phase="feature-sets",
                model_id="feature_set_ridge",
                model_name="Ridge",
                model_family="linear",
                feature_set=feature_set,
                preprocessing_family="linear",
                hyperparameters={"alpha": 1.0, "solver": "auto"},
            )
        )
        configs.append(
            BenchmarkConfig(
                model_run_id=f"feature_set_decision_tree_{feature_set}",
                phase="feature-sets",
                model_id="feature_set_decision_tree",
                model_name="DecisionTree",
                model_family="tree",
                feature_set=feature_set,
                preprocessing_family="tree",
                hyperparameters={"max_depth": 15, "min_samples_leaf": 10},
            )
        )
    return configs


def real_model_configs(selected_feature_set: str, random_state: int = 42) -> list[BenchmarkConfig]:
    configs = [
        BenchmarkConfig(
            model_run_id=f"linear_regression_{selected_feature_set}",
            phase="models",
            model_id="linear_regression",
            model_name="LinearRegression",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"fit_intercept": True, "n_jobs": -1},
        ),
        BenchmarkConfig(
            model_run_id=f"ridge_alpha_0_1_{selected_feature_set}",
            phase="models",
            model_id="ridge_alpha_0_1",
            model_name="Ridge",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"alpha": 0.1, "solver": "auto"},
        ),
        BenchmarkConfig(
            model_run_id=f"ridge_alpha_1_{selected_feature_set}",
            phase="models",
            model_id="ridge_alpha_1",
            model_name="Ridge",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"alpha": 1.0, "solver": "auto"},
        ),
        BenchmarkConfig(
            model_run_id=f"ridge_alpha_10_{selected_feature_set}",
            phase="models",
            model_id="ridge_alpha_10",
            model_name="Ridge",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"alpha": 10.0, "solver": "auto"},
        ),
        BenchmarkConfig(
            model_run_id=f"lasso_alpha_0_0001_{selected_feature_set}",
            phase="models",
            model_id="lasso_alpha_0_0001",
            model_name="Lasso",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"alpha": 0.0001, "max_iter": 5000, "selection": "cyclic"},
        ),
        BenchmarkConfig(
            model_run_id=f"lasso_alpha_0_001_{selected_feature_set}",
            phase="models",
            model_id="lasso_alpha_0_001",
            model_name="Lasso",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"alpha": 0.001, "max_iter": 5000, "selection": "cyclic"},
        ),
        BenchmarkConfig(
            model_run_id=f"tree_depth_8_leaf_20_{selected_feature_set}",
            phase="models",
            model_id="tree_depth_8_leaf_20",
            model_name="DecisionTree",
            model_family="tree",
            feature_set=selected_feature_set,
            preprocessing_family="tree",
            hyperparameters={"max_depth": 8, "min_samples_leaf": 20},
        ),
        BenchmarkConfig(
            model_run_id=f"tree_depth_15_leaf_10_{selected_feature_set}",
            phase="models",
            model_id="tree_depth_15_leaf_10",
            model_name="DecisionTree",
            model_family="tree",
            feature_set=selected_feature_set,
            preprocessing_family="tree",
            hyperparameters={"max_depth": 15, "min_samples_leaf": 10},
        ),
        BenchmarkConfig(
            model_run_id=f"tree_depth_none_leaf_20_{selected_feature_set}",
            phase="models",
            model_id="tree_depth_none_leaf_20",
            model_name="DecisionTree",
            model_family="tree",
            feature_set=selected_feature_set,
            preprocessing_family="tree",
            hyperparameters={"max_depth": None, "min_samples_leaf": 20},
        ),
        BenchmarkConfig(
            model_run_id=f"rf_150_depth_20_leaf_5_{selected_feature_set}",
            phase="models",
            model_id="rf_150_depth_20_leaf_5",
            model_name="RandomForest",
            model_family="tree",
            feature_set=selected_feature_set,
            preprocessing_family="tree",
            hyperparameters={
                "n_estimators": 150,
                "max_depth": 20,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        ),
        BenchmarkConfig(
            model_run_id=f"rf_200_depth_none_leaf_10_{selected_feature_set}",
            phase="models",
            model_id="rf_200_depth_none_leaf_10",
            model_name="RandomForest",
            model_family="tree",
            feature_set=selected_feature_set,
            preprocessing_family="tree",
            hyperparameters={
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        ),
        BenchmarkConfig(
            model_run_id=f"linearsvr_c_0_1_{selected_feature_set}",
            phase="models",
            model_id="linearsvr_c_0_1",
            model_name="LinearSVR",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"C": 0.1},
        ),
        BenchmarkConfig(
            model_run_id=f"linearsvr_c_1_{selected_feature_set}",
            phase="models",
            model_id="linearsvr_c_1",
            model_name="LinearSVR",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"C": 1.0},
        ),
        BenchmarkConfig(
            model_run_id=f"linearsvr_c_10_{selected_feature_set}",
            phase="models",
            model_id="linearsvr_c_10",
            model_name="LinearSVR",
            model_family="linear",
            feature_set=selected_feature_set,
            preprocessing_family="linear",
            hyperparameters={"C": 10.0},
        ),
        BenchmarkConfig(
            model_run_id=f"knn_k_5_{selected_feature_set}",
            phase="models",
            model_id="knn_k_5",
            model_name="KNN",
            model_family="knn",
            feature_set=selected_feature_set,
            preprocessing_family="knn",
            hyperparameters={"n_neighbors": 5, "weights": "distance", "n_jobs": -1},
            training_scope="resource_limited_15000_train_rows",
        ),
        BenchmarkConfig(
            model_run_id=f"knn_k_15_{selected_feature_set}",
            phase="models",
            model_id="knn_k_15",
            model_name="KNN",
            model_family="knn",
            feature_set=selected_feature_set,
            preprocessing_family="knn",
            hyperparameters={"n_neighbors": 15, "weights": "distance", "n_jobs": -1},
            training_scope="resource_limited_15000_train_rows",
        ),
    ]
    return configs


def successful_phase_runs(phase: str, path: Path = RUNS_PATH) -> pd.DataFrame:
    runs = read_runs(path)
    return runs[runs["phase"].astype(str).eq(phase) & runs["status"].astype(str).eq("completed")].copy()


def write_baseline_report(path: Path = BASELINE_REPORT_PATH, runs_path: Path = RUNS_PATH) -> pd.DataFrame:
    baselines = successful_phase_runs("baselines", runs_path)
    if baselines.empty:
        raise ValueError("No completed baseline runs found")
    baselines = baselines.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    ensure_parent(path)
    baselines.to_csv(path, index=False)
    return baselines


def select_feature_set(comparison: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    required_models = {"Ridge", "DecisionTree"}
    completed = comparison[comparison["status"].astype(str).eq("completed")].copy()
    if set(completed["model_name"]) != required_models:
        raise ValueError("Feature-set phase needs completed Ridge and DecisionTree anchor results")
    completed["mae_rank"] = completed.groupby("model_name")["mae"].rank(method="average", ascending=True)
    summary = (
        completed.groupby("feature_set", as_index=False)
        .agg(average_mae_rank=("mae_rank", "mean"), average_mae=("mae", "mean"))
        .copy()
    )
    summary["simplicity_rank"] = summary["feature_set"].map({"core_without_lag": 0, "core_with_lag": 1}).fillna(2)
    summary = summary.sort_values(["average_mae_rank", "average_mae", "simplicity_rank"], ascending=[True, True, True])
    return str(summary.iloc[0]["feature_set"]), completed


def write_feature_set_selection(
    path: Path = FEATURE_SET_REPORT_PATH,
    selected_path: Path = SELECTED_FEATURE_SET_PATH,
    runs_path: Path = RUNS_PATH,
) -> tuple[str, pd.DataFrame]:
    phase_rows = read_runs(runs_path)
    phase_rows = phase_rows[phase_rows["phase"].astype(str).eq("feature-sets")].copy()
    if phase_rows.empty:
        raise ValueError("No feature-set comparison results found")
    selected_feature_set, ranked = select_feature_set(phase_rows)
    output = phase_rows.merge(
        ranked[["model_run_id", "mae_rank"]],
        on="model_run_id",
        how="left",
    )
    output["selected_feature_set"] = selected_feature_set
    ensure_parent(path)
    output.to_csv(path, index=False)

    selection = {
        "selected_feature_set": selected_feature_set,
        "selection_metric": "lowest average validation MAE rank across Ridge and DecisionTree; tie-break by average MAE, then core_without_lag",
        "anchor_models": ["Ridge", "DecisionTree"],
        "comparison_results": ranked.sort_values(["model_name", "feature_set"]).to_dict(orient="records"),
        "lag_interpretation": LAG_INTERPRETATION,
        "created_at_utc": utc_now(),
    }
    write_json(selected_path, selection)
    return selected_feature_set, output


def load_selected_feature_set(path: Path = SELECTED_FEATURE_SET_PATH) -> str:
    if not path.exists():
        raise ValueError("selected_validation_feature_set.json does not exist; run --phase feature-sets first")
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected_feature_set")
    if selected not in {"core_without_lag", "core_with_lag"}:
        raise ValueError(f"Invalid selected feature set: {selected}")
    return str(selected)


def model_family_rank(name: str) -> int:
    try:
        return SIMPLE_MODEL_ORDER.index(str(name))
    except ValueError:
        return len(SIMPLE_MODEL_ORDER)


def select_winning_model(model_results: pd.DataFrame) -> pd.Series:
    completed = model_results[
        model_results["phase"].astype(str).eq("models")
        & model_results["status"].astype(str).eq("completed")
    ].copy()
    if completed.empty:
        raise ValueError("No successful real model runs available for winner selection")
    completed["model_family_rank"] = completed["model_name"].map(model_family_rank)
    completed = completed.sort_values(
        ["mae", "rmse", "r2", "model_family_rank"],
        ascending=[True, True, False, True],
    )
    return completed.iloc[0]


def best_baseline(baseline_results: pd.DataFrame) -> pd.Series:
    completed = baseline_results[baseline_results["status"].astype(str).eq("completed")].copy()
    if completed.empty:
        raise ValueError("No successful baseline results")
    return completed.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).iloc[0]


def load_predictions_for_run(model_run_id: str, predictions_dir: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    path = predictions_dir / f"{model_run_id}.csv"
    if not path.exists():
        raise ValueError(f"Missing validation predictions for selected run: {path}")
    return pd.read_csv(path)


def compute_subgroup_metrics(validation: pd.DataFrame, predictions: pd.DataFrame, selected_model_run_id: str) -> pd.DataFrame:
    merged = validation.merge(predictions, on="canonical_crop_row_id", how="inner", validate="one_to_one")
    if len(merged) != len(validation):
        raise ValueError("Prediction rows do not match validation rows")
    merged["absolute_error"] = (merged[TARGET_COLUMN] - merged["prediction"]).abs()
    rows = []
    for column in ["Crop_canonical", "Season_canonical", "canonical_state_name", "Crop_Year"]:
        grouped = merged.groupby(column, dropna=False)
        for value, group in grouped:
            rows.append(
                {
                    "selected_model_run_id": selected_model_run_id,
                    "subgroup_column": column,
                    "subgroup_value": value,
                    "row_count": len(group),
                    "mae": float(group["absolute_error"].mean()),
                    "reliable_group": len(group) >= 20,
                }
            )
    return pd.DataFrame(rows)


def make_prediction_sample(validation: pd.DataFrame, predictions: pd.DataFrame, selected_model_run_id: str, max_rows: int = 500) -> pd.DataFrame:
    merged = validation.merge(predictions, on="canonical_crop_row_id", how="inner", validate="one_to_one")
    merged["absolute_error"] = (merged[TARGET_COLUMN] - merged["prediction"]).abs()
    sample = merged.sort_values("canonical_crop_row_id").head(max_rows).copy()
    sample["selected_model_run_id"] = selected_model_run_id
    columns = [
        "canonical_crop_row_id",
        "Crop_Year",
        "canonical_state_name",
        "canonical_district_name",
        "Crop_canonical",
        "Season_canonical",
        TARGET_COLUMN,
        "prediction",
        "absolute_error",
        "selected_model_run_id",
    ]
    return sample[columns]


def plot_metric(results: pd.DataFrame, metric: str, path: Path) -> None:
    successful = results[results["status"].astype(str).eq("completed")].copy()
    if successful.empty:
        raise ValueError(f"No successful results to plot for {metric}")
    successful["label"] = successful["model_run_id"].astype(str)
    successful = successful.sort_values(metric, ascending=(metric != "r2"))
    plt.figure(figsize=(12, max(5, len(successful) * 0.35)))
    plt.barh(successful["label"], successful[metric])
    plt.xlabel(metric.upper())
    plt.ylabel("model_run_id")
    plt.title(f"Validation {metric.upper()}")
    plt.tight_layout()
    ensure_parent(path)
    plt.savefig(path, dpi=150)
    plt.close()


def parse_hyperparameters(row: pd.Series) -> dict[str, Any]:
    raw = row.get("hyperparameters_json", "{}")
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def build_frozen_configuration(
    selected_model: pd.Series,
    baseline_rows: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    return {
        "selected_model_run_id": selected_model["model_run_id"],
        "selected_model_name": selected_model["model_name"],
        "selected_feature_set": selected_model["feature_set"],
        "selected_hyperparameters": parse_hyperparameters(selected_model),
        "selected_preprocessing_family": selected_model["preprocessing_family"],
        "target_column": TARGET_COLUMN,
        "train_period": "1997-2010",
        "validation_period": "2011-2012",
        "test_period": "2013-2014",
        "selection_metric": "validation MAE; tie-break RMSE, R2, simpler model family",
        "tie_break_policy": [
            "lowest MAE",
            "lowest RMSE",
            "highest R2",
            "simpler model: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, LinearSVR, KNN",
        ],
        "baseline_results": baseline_rows.to_dict(orient="records"),
        "validation_metrics": {
            "mae": float(selected_model["mae"]),
            "rmse": float(selected_model["rmse"]),
            "r2": float(selected_model["r2"]),
            "median_ae": float(selected_model["median_ae"]),
        },
        "training_scope": selected_model["training_scope"],
        "random_state": random_state,
        "forbidden_test_usage": FORBIDDEN_TEST_USAGE,
        "test_data_accessed": False,
        "created_at_utc": utc_now(),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
    display = frame.copy().fillna("")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_summary(
    all_successful: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    feature_rows: pd.DataFrame,
    selected_model: pd.Series,
    selected_feature_set: str,
    best_baseline_row: pd.Series,
) -> str:
    improvement = float(best_baseline_row["mae"]) - float(selected_model["mae"])
    relative = improvement / float(best_baseline_row["mae"]) * 100 if float(best_baseline_row["mae"]) != 0 else np.nan
    warnings_frame = all_successful[pd.to_numeric(all_successful["warning_count"], errors="coerce").fillna(0).gt(0)]
    failed = read_runs()
    failed = failed[failed["status"].astype(str).eq("failed")]
    model_table = markdown_table(
        all_successful.sort_values(["phase", "mae"])[
            ["model_run_id", "phase", "model_name", "feature_set", "mae", "rmse", "r2", "training_scope"]
        ]
    )
    baseline_table = markdown_table(baseline_rows[["model_run_id", "mae", "rmse", "r2", "median_ae"]])
    feature_table = markdown_table(feature_rows[["model_run_id", "feature_set", "model_name", "mae", "rmse", "r2"]])
    warning_text = markdown_table(warnings_frame[["model_run_id", "warning_summary"]])
    failed_text = markdown_table(failed[["model_run_id", "error_type", "error_message"]])
    return "\n".join(
        [
            "# Validation Benchmark Summary",
            "",
            f"- Train rows: {EXPECTED_TRAIN_ROWS}",
            f"- Validation rows: {EXPECTED_VALIDATION_ROWS}",
            "- Test data accessed: false",
            "- Final test evaluation: not performed",
            f"- Selected feature set: `{selected_feature_set}`",
            f"- Best model: `{selected_model['model_run_id']}`",
            f"- Best baseline: `{best_baseline_row['model_run_id']}`",
            f"- Absolute MAE improvement over baseline: {improvement:.6f}",
            f"- Relative MAE improvement over baseline: {relative:.2f}%",
            "",
            "Overfitting cannot be determined definitively from validation results alone.",
            "KNN uses resource-limited training with at most 15,000 train rows, so it is not fully directly comparable to full-train models.",
            "",
            "## Baselines",
            "",
            baseline_table,
            "",
            "## Feature Set Comparison",
            "",
            feature_table,
            "",
            "## All Successful Models",
            "",
            model_table,
            "",
            "## Warnings",
            "",
            warning_text,
            "",
            "## Failed Runs",
            "",
            failed_text,
            "",
            "The 2013-2014 test split has not been used for preprocessing, feature selection, hyperparameter tuning, model selection or final evaluation.",
            "",
        ]
    )


def finalize_outputs(random_state: int = 42) -> pd.Series:
    runs = read_runs()
    successful = runs[runs["status"].astype(str).eq("completed")].copy()
    if successful.empty:
        raise ValueError("No successful benchmark runs found")
    baseline_rows = successful[successful["phase"].astype(str).eq("baselines")].copy()
    feature_rows = successful[successful["phase"].astype(str).eq("feature-sets")].copy()
    if baseline_rows.empty:
        raise ValueError("Baseline phase results are required before finalize")
    if feature_rows.empty:
        raise ValueError("Feature-set phase results are required before finalize")
    selected_feature_set = load_selected_feature_set()
    selected_model = select_winning_model(successful)
    baseline_winner = best_baseline(baseline_rows)

    ensure_parent(MODEL_RESULTS_PATH)
    successful.sort_values(["phase", "mae"]).to_csv(MODEL_RESULTS_PATH, index=False)
    runtime_columns = [
        "model_run_id",
        "phase",
        "model_name",
        "training_scope",
        "train_rows_used",
        "validation_rows",
        "fit_seconds",
        "predict_seconds",
        "status",
        "warning_count",
    ]
    runs[runtime_columns].to_csv(RUNTIME_RESULTS_PATH, index=False)

    validation = safe_read_parquet(VALIDATION_PATH)
    predictions = load_predictions_for_run(str(selected_model["model_run_id"]))
    subgroup = compute_subgroup_metrics(validation, predictions, str(selected_model["model_run_id"]))
    subgroup.to_csv(SUBGROUP_METRICS_PATH, index=False)
    sample = make_prediction_sample(validation, predictions, str(selected_model["model_run_id"]))
    sample.to_csv(PREDICTIONS_SAMPLE_PATH, index=False)

    plot_rows = successful[successful["phase"].isin(["baselines", "models"])].copy()
    plot_metric(plot_rows, "mae", MAE_PLOT_PATH)
    plot_metric(plot_rows, "rmse", RMSE_PLOT_PATH)
    plot_metric(plot_rows, "r2", R2_PLOT_PATH)

    frozen = build_frozen_configuration(selected_model, baseline_rows, random_state)
    write_json(FROZEN_MODEL_CONFIG_PATH, frozen)

    summary = make_summary(successful, baseline_rows, feature_rows, selected_model, selected_feature_set, baseline_winner)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    return selected_model


def run_phase(phase: str, resume: bool, force_model: str | None, random_state: int) -> None:
    if phase == "finalize":
        selected_model = finalize_outputs(random_state=random_state)
        print(f"selected_model_run_id={selected_model['model_run_id']}")
        print(f"selected_feature_set={selected_model['feature_set']}")
        print(f"validation_mae={selected_model['mae']}")
        return

    train, validation, manifest = load_benchmark_inputs()
    if phase == "baselines":
        configs = baseline_configs()
        run_config_sequence(configs, train, validation, manifest, resume, force_model, random_state)
        baselines = write_baseline_report()
        lag_rows = int(validation["lag_available"].astype(int).eq(1).sum()) if "lag_available" in validation else 0
        winner = best_baseline(baselines)
        print(f"validation_rows_with_lag={lag_rows}")
        print(f"best_baseline={winner['model_run_id']}")
        print(f"best_baseline_mae={winner['mae']}")
        return
    if phase == "feature-sets":
        configs = feature_set_configs(random_state=random_state)
        run_config_sequence(configs, train, validation, manifest, resume, force_model, random_state)
        selected, comparison = write_feature_set_selection()
        print(f"selected_feature_set={selected}")
        for model_name in sorted(comparison["model_name"].dropna().unique()):
            subset = comparison[comparison["model_name"].eq(model_name)]
            pivot = subset.pivot_table(index="model_name", columns="feature_set", values="mae", aggfunc="first")
            if {"core_without_lag", "core_with_lag"}.issubset(set(pivot.columns)):
                delta = float(pivot["core_with_lag"].iloc[0] - pivot["core_without_lag"].iloc[0])
                print(f"{model_name}_mae_delta_core_with_lag_minus_without={delta}")
        return
    if phase == "models":
        selected = load_selected_feature_set()
        configs = real_model_configs(selected, random_state=random_state)
        run_config_sequence(configs, train, validation, manifest, resume, force_model, random_state)
        model_rows = read_runs()
        model_rows = model_rows[model_rows["phase"].astype(str).eq("models")]
        completed = int(model_rows["status"].astype(str).eq("completed").sum())
        failed = int(model_rows["status"].astype(str).eq("failed").sum())
        print(f"selected_feature_set={selected}")
        print(f"successful_model_runs={completed}")
        print(f"failed_model_runs={failed}")
        if completed:
            winner = select_winning_model(model_rows)
            print(f"current_best_model={winner['model_run_id']}")
            print(f"current_best_mae={winner['mae']}")
        return
    raise ValueError(f"Unsupported phase: {phase}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation benchmark phases without touching the test split.")
    parser.add_argument("--phase", choices=["baselines", "feature-sets", "models", "finalize"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-model", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_phase(
            phase=args.phase,
            resume=bool(args.resume),
            force_model=args.force_model,
            random_state=int(args.random_state),
        )
    except Exception as error:
        print(f"Validation benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
