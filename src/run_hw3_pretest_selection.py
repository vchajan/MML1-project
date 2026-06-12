from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/train_1997_2010.parquet"
VALIDATION_PATH = ROOT / "data/processed/validation_2011_2012.parquet"
MANIFEST_PATH = ROOT / "data/reference/model_feature_manifest.json"

# These files must remain outside the repository during pre-test selection.
HOLDOUT_TEST_PATH = ROOT / "data/processed/test_2013_2014.parquet"
FULL_MODEL_DATASET_PATH = ROOT / "data/processed/model_dataset_1997_2014.parquet"

FULL_RESULTS_PATH = ROOT / "reports/hw3_pretest_full_validation_results.csv"
RESOURCE_RESULTS_PATH = ROOT / "reports/hw3_pretest_resource_validation_results.csv"
SUMMARY_PATH = ROOT / "reports/hw3_pretest_selection_summary.md"
FROZEN_PLAN_PATH = ROOT / "data/reference/hw3_frozen_evaluation_plan.json"

RANDOM_STATE = 42
TARGET = "target_yield"
FEATURE_SET_NAME = "core_without_lag"
FULL_SCOPE = "full_train_validation"
RESOURCE_SCOPE = "resource_limited_train_validation"

DEFAULT_RESOURCE_TRAIN_ROWS = 5000
DEFAULT_RESOURCE_VALIDATION_ROWS = 2000

FULL_REQUIRED_RUNS = [
    "dummy_mean",
    "dummy_median",
    "linear_regression",
    "ridge_alpha_1",
    "lasso_alpha_0_01",
    "elastic_net_alpha_0_01_l1_0_5",
    "decision_tree_leaf_20",
    "random_forest_200_leaf_20",
    "gradient_boosting_60_lr_0_1_depth_3",
    "linear_svr_c_0_03",
]

RESOURCE_REQUIRED_RUNS = [
    "sample_dummy_mean",
    "sample_dummy_median",
    "sample_decision_tree_leaf_20",
    "sample_knn_15_distance",
    "sample_svr_rbf_c_10_epsilon_5",
]

RESULT_COLUMNS = [
    "model_run_id",
    "evaluation_scope",
    "model_name",
    "model_family",
    "feature_set",
    "preprocessing_family",
    "hyperparameters_json",
    "fit_period",
    "evaluation_period",
    "train_rows_available",
    "train_rows_used",
    "validation_rows_available",
    "validation_rows_used",
    "fit_seconds",
    "train_predict_seconds",
    "validation_predict_seconds",
    "train_mae",
    "train_rmse",
    "train_r2",
    "train_median_ae",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "validation_median_ae",
    "mae_generalization_gap",
    "rmse_generalization_gap",
    "r2_generalization_gap",
    "status",
    "warning_count",
    "warning_summary",
    "error_type",
    "error_message",
    "random_state",
]


@dataclass(frozen=True)
class ModelSpec:
    run_id: str
    model_name: str
    model_family: str
    preprocessing_family: str
    hyperparameters: dict
    build_estimator: Callable[[], object]


def assert_holdout_absent() -> None:
    forbidden_present = [
        path for path in (HOLDOUT_TEST_PATH, FULL_MODEL_DATASET_PATH) if path.exists()
    ]
    if forbidden_present:
        joined = ", ".join(str(path) for path in forbidden_present)
        raise RuntimeError(
            "Pre-test selection refuses to run while holdout-containing files "
            f"are present in the repository: {joined}"
        )


def one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "min_frequency": 5}
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=True, **kwargs)


def build_linear_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "median_ae": float(median_absolute_error(y, pred)),
    }


def deterministic_sample(
    frame: pd.DataFrame,
    n_rows: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if len(frame) <= n_rows:
        return frame.copy()
    return (
        frame.sample(n=n_rows, random_state=random_state)
        .sort_index()
        .copy()
    )


def load_pretest_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    assert_holdout_absent()

    required = [TRAIN_PATH, VALIDATION_PATH, MANIFEST_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required pre-test inputs: {missing}")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    train_years = set(int(year) for year in train["Crop_Year"].unique())
    validation_years = set(
        int(year) for year in validation["Crop_Year"].unique()
    )
    if train_years != set(range(1997, 2011)):
        raise ValueError(f"Unexpected train years: {sorted(train_years)}")
    if validation_years != {2011, 2012}:
        raise ValueError(
            f"Unexpected validation years: {sorted(validation_years)}"
        )

    feature_set = list(manifest["feature_sets"][FEATURE_SET_NAME])
    forbidden = set(manifest["forbidden_leakage_columns"])
    leakage = sorted(set(feature_set) & forbidden)
    if leakage:
        raise ValueError(f"Leakage columns in feature set: {leakage}")

    lag_columns = {"lag_yield_1y", "lag_available"}
    if lag_columns & set(feature_set):
        raise ValueError(
            "The common HW3 comparison must use the no-lag feature set."
        )

    if train[TARGET].isna().any() or validation[TARGET].isna().any():
        raise ValueError("Missing target values are not allowed.")

    return train, validation, manifest


def full_model_specs(
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[ModelSpec]:
    linear = lambda: build_linear_preprocessor(
        categorical_features, numeric_features
    )
    tree = lambda: build_tree_preprocessor(
        categorical_features, numeric_features
    )

    return [
        ModelSpec(
            "dummy_mean",
            "DummyRegressor (mean)",
            "baseline",
            "none",
            {"strategy": "mean"},
            lambda: DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            "dummy_median",
            "DummyRegressor (median)",
            "baseline",
            "none",
            {"strategy": "median"},
            lambda: DummyRegressor(strategy="median"),
        ),
        ModelSpec(
            "linear_regression",
            "Linear Regression",
            "linear",
            "onehot_scaled",
            {},
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        ModelSpec(
            "ridge_alpha_1",
            "Ridge",
            "linear_regularized",
            "onehot_scaled",
            {"alpha": 1.0, "solver": "lsqr"},
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    ("model", Ridge(alpha=1.0, solver="lsqr")),
                ]
            ),
        ),
        ModelSpec(
            "lasso_alpha_0_01",
            "Lasso",
            "linear_regularized",
            "onehot_scaled",
            {
                "alpha": 0.01,
                "max_iter": 10000,
                "tol": 0.001,
                "selection": "cyclic",
            },
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    (
                        "model",
                        Lasso(
                            alpha=0.01,
                            max_iter=10000,
                            tol=0.001,
                            selection="cyclic",
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "elastic_net_alpha_0_01_l1_0_5",
            "Elastic Net",
            "linear_regularized",
            "onehot_scaled",
            {
                "alpha": 0.01,
                "l1_ratio": 0.5,
                "max_iter": 10000,
                "tol": 0.001,
                "selection": "cyclic",
            },
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.01,
                            l1_ratio=0.5,
                            max_iter=10000,
                            tol=0.001,
                            selection="cyclic",
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "decision_tree_leaf_20",
            "Decision Tree",
            "tree",
            "ordinal_unscaled",
            {
                "max_depth": None,
                "min_samples_leaf": 20,
                "random_state": RANDOM_STATE,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", tree()),
                    (
                        "model",
                        DecisionTreeRegressor(
                            max_depth=None,
                            min_samples_leaf=20,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "random_forest_200_leaf_20",
            "Random Forest",
            "bagging_ensemble",
            "ordinal_unscaled",
            {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 20,
                "max_features": 0.5,
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", tree()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            max_depth=None,
                            min_samples_leaf=20,
                            max_features=0.5,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "gradient_boosting_60_lr_0_1_depth_3",
            "Gradient Boosting",
            "boosting_ensemble",
            "ordinal_unscaled",
            {
                "n_estimators": 60,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": RANDOM_STATE,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", tree()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=60,
                            learning_rate=0.1,
                            max_depth=3,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "linear_svr_c_0_03",
            "LinearSVR",
            "svm",
            "onehot_scaled",
            {
                "C": 0.03,
                "epsilon": 0.0,
                "max_iter": 20000,
                "dual": "auto",
                "random_state": RANDOM_STATE,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    (
                        "model",
                        LinearSVR(
                            C=0.03,
                            epsilon=0.0,
                            max_iter=20000,
                            dual="auto",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]


def resource_model_specs(
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[ModelSpec]:
    linear = lambda: build_linear_preprocessor(
        categorical_features, numeric_features
    )
    tree = lambda: build_tree_preprocessor(
        categorical_features, numeric_features
    )

    return [
        ModelSpec(
            "sample_dummy_mean",
            "DummyRegressor mean (sample)",
            "baseline",
            "none",
            {"strategy": "mean"},
            lambda: DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            "sample_dummy_median",
            "DummyRegressor median (sample)",
            "baseline",
            "none",
            {"strategy": "median"},
            lambda: DummyRegressor(strategy="median"),
        ),
        ModelSpec(
            "sample_decision_tree_leaf_20",
            "Decision Tree (sample)",
            "tree",
            "ordinal_unscaled",
            {
                "max_depth": None,
                "min_samples_leaf": 20,
                "random_state": RANDOM_STATE,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", tree()),
                    (
                        "model",
                        DecisionTreeRegressor(
                            max_depth=None,
                            min_samples_leaf=20,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "sample_knn_15_distance",
            "KNN (sample)",
            "neighbors",
            "onehot_scaled",
            {
                "n_neighbors": 15,
                "weights": "distance",
                "algorithm": "brute",
                "n_jobs": -1,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    (
                        "model",
                        KNeighborsRegressor(
                            n_neighbors=15,
                            weights="distance",
                            algorithm="brute",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "sample_svr_rbf_c_10_epsilon_5",
            "SVR RBF (sample)",
            "kernel_svm",
            "onehot_scaled",
            {
                "kernel": "rbf",
                "C": 10.0,
                "gamma": "scale",
                "epsilon": 5.0,
            },
            lambda: Pipeline(
                [
                    ("preprocessor", linear()),
                    (
                        "model",
                        SVR(
                            kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            epsilon=5.0,
                        ),
                    ),
                ]
            ),
        ),
    ]


def empty_result(
    spec: ModelSpec,
    scope: str,
    train_rows_available: int,
    train_rows_used: int,
    validation_rows_available: int,
    validation_rows_used: int,
) -> dict:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "model_run_id": spec.run_id,
            "evaluation_scope": scope,
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "feature_set": FEATURE_SET_NAME,
            "preprocessing_family": spec.preprocessing_family,
            "hyperparameters_json": json.dumps(
                spec.hyperparameters, sort_keys=True
            ),
            "fit_period": "1997-2010",
            "evaluation_period": "2011-2012",
            "train_rows_available": train_rows_available,
            "train_rows_used": train_rows_used,
            "validation_rows_available": validation_rows_available,
            "validation_rows_used": validation_rows_used,
            "status": "pending",
            "warning_count": 0,
            "warning_summary": "",
            "error_type": "",
            "error_message": "",
            "random_state": RANDOM_STATE,
        }
    )
    return row


def read_scope_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[RESULT_COLUMNS]


def write_scope_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_scope_results(path)
    frame = frame[frame["model_run_id"] != row["model_run_id"]]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame[RESULT_COLUMNS].sort_values("model_run_id").to_csv(
        path, index=False
    )


def completed_run_ids(path: Path) -> set[str]:
    frame = read_scope_results(path)
    return set(
        frame.loc[frame["status"].eq("completed"), "model_run_id"].astype(str)
    )


def run_one_model(
    spec: ModelSpec,
    scope: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_set: list[str],
    train_rows_available: int,
    validation_rows_available: int,
) -> dict:
    row = empty_result(
        spec,
        scope,
        train_rows_available,
        len(train_frame),
        validation_rows_available,
        len(validation_frame),
    )

    estimator = spec.build_estimator()
    X_train = train_frame[feature_set]
    y_train = train_frame[TARGET]
    X_validation = validation_frame[feature_set]
    y_validation = validation_frame[TARGET]

    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            started = time.perf_counter()
            if spec.preprocessing_family == "none":
                estimator.fit(np.zeros((len(train_frame), 1)), y_train)
            else:
                estimator.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            if spec.preprocessing_family == "none":
                train_prediction = estimator.predict(
                    np.zeros((len(train_frame), 1))
                )
            else:
                train_prediction = estimator.predict(X_train)
            train_predict_seconds = time.perf_counter() - started

            started = time.perf_counter()
            if spec.preprocessing_family == "none":
                validation_prediction = estimator.predict(
                    np.zeros((len(validation_frame), 1))
                )
            else:
                validation_prediction = estimator.predict(X_validation)
            validation_predict_seconds = time.perf_counter() - started

            warning_messages = [
                f"{item.category.__name__}: {item.message}" for item in caught
            ]

        train_metrics = regression_metrics(y_train, train_prediction)
        validation_metrics = regression_metrics(
            y_validation, validation_prediction
        )

        row.update(
            {
                "fit_seconds": fit_seconds,
                "train_predict_seconds": train_predict_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_median_ae": train_metrics["median_ae"],
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "validation_median_ae": validation_metrics["median_ae"],
                "mae_generalization_gap": (
                    validation_metrics["mae"] - train_metrics["mae"]
                ),
                "rmse_generalization_gap": (
                    validation_metrics["rmse"] - train_metrics["rmse"]
                ),
                "r2_generalization_gap": (
                    validation_metrics["r2"] - train_metrics["r2"]
                ),
                "status": "completed",
                "warning_count": len(warning_messages),
                "warning_summary": " | ".join(warning_messages)[:4000],
            }
        )
    except Exception as exc:
        traceback.print_exc()
        row.update(
            {
                "status": "failed",
                "warning_count": len(warning_messages),
                "warning_summary": " | ".join(warning_messages)[:4000],
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
            }
        )

    return row


def run_phase(
    phase: str,
    resume: bool,
    force_run: str | None,
    resource_train_rows: int,
    resource_validation_rows: int,
) -> None:
    train, validation, manifest = load_pretest_inputs()
    feature_set = list(manifest["feature_sets"][FEATURE_SET_NAME])
    categorical = list(manifest["categorical_features"])
    numeric = [
        column for column in feature_set if column not in categorical
    ]

    if phase == "full-validation":
        train_frame = train
        validation_frame = validation
        specs = full_model_specs(categorical, numeric)
        results_path = FULL_RESULTS_PATH
        scope = FULL_SCOPE
    elif phase == "resource-validation":
        train_frame = deterministic_sample(
            train, resource_train_rows, RANDOM_STATE
        )
        validation_frame = deterministic_sample(
            validation, resource_validation_rows, RANDOM_STATE
        )
        specs = resource_model_specs(categorical, numeric)
        results_path = RESOURCE_RESULTS_PATH
        scope = RESOURCE_SCOPE
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    if force_run:
        specs = [spec for spec in specs if spec.run_id == force_run]
        if not specs:
            raise ValueError(f"Unknown run ID for {phase}: {force_run}")

    completed = completed_run_ids(results_path)
    for spec in specs:
        if resume and spec.run_id in completed and force_run is None:
            print(f"model_run_id={spec.run_id} status=skipped_resume")
            continue

        row = run_one_model(
            spec=spec,
            scope=scope,
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_set=feature_set,
            train_rows_available=len(train),
            validation_rows_available=len(validation),
        )
        write_scope_result(results_path, row)
        print(
            f"model_run_id={spec.run_id} "
            f"status={row['status']} "
            f"validation_mae={row.get('validation_mae')}"
        )


def require_completed(
    frame: pd.DataFrame,
    required_run_ids: list[str],
    label: str,
) -> None:
    completed = set(
        frame.loc[frame["status"].eq("completed"), "model_run_id"].astype(str)
    )
    missing = [run_id for run_id in required_run_ids if run_id not in completed]
    failed = frame.loc[frame["status"].eq("failed"), "model_run_id"].tolist()
    if missing or failed:
        raise RuntimeError(
            f"Cannot freeze {label}. Missing={missing}; failed={failed}"
        )


def specs_to_plan(specs: list[ModelSpec]) -> list[dict]:
    return [
        {
            "run_id": spec.run_id,
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "preprocessing_family": spec.preprocessing_family,
            "feature_set": FEATURE_SET_NAME,
            "hyperparameters": spec.hyperparameters,
        }
        for spec in specs
    ]



def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []

    for values in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{float_digits}f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)

    widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in rows),
        )
        for index in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ) + " |"

    header_row = render_row(headers)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = "\n".join(render_row(row) for row in rows)
    return "\n".join(part for part in [header_row, separator, body] if part)


def build_summary(
    full: pd.DataFrame,
    resource: pd.DataFrame,
) -> str:
    ordered = full.sort_values(
        ["validation_mae", "validation_rmse", "model_run_id"]
    )
    best = ordered.iloc[0]
    mean_baseline = full.loc[
        full["model_run_id"].eq("dummy_mean")
    ].iloc[0]
    median_baseline = full.loc[
        full["model_run_id"].eq("dummy_median")
    ].iloc[0]

    absolute_mean_gain = (
        float(mean_baseline["validation_mae"])
        - float(best["validation_mae"])
    )
    relative_mean_gain = (
        absolute_mean_gain / float(mean_baseline["validation_mae"]) * 100.0
    )
    absolute_median_gain = (
        float(median_baseline["validation_mae"])
        - float(best["validation_mae"])
    )
    relative_median_gain = (
        absolute_median_gain
        / float(median_baseline["validation_mae"])
        * 100.0
    )

    resource_ordered = resource.sort_values(
        ["validation_mae", "validation_rmse", "model_run_id"]
    )
    resource_summary = resource_ordered.loc[
        resource_ordered["model_run_id"].isin(RESOURCE_REQUIRED_RUNS)
    ].copy()

    present_resource_ids = set(
        resource_summary["model_run_id"].astype(str)
    )
    missing_resource_ids = [
        run_id
        for run_id in RESOURCE_REQUIRED_RUNS
        if run_id not in present_resource_ids
    ]
    if missing_resource_ids:
        raise RuntimeError(
            "Cannot build pre-test summary. Missing resource models: "
            + ", ".join(missing_resource_ids)
        )

    if len(resource_summary) != len(RESOURCE_REQUIRED_RUNS):
        raise RuntimeError(
            "Cannot build pre-test summary. Expected "
            f"{len(RESOURCE_REQUIRED_RUNS)} resource rows, found "
            f"{len(resource_summary)}."
        )

    return f"""# HW3 – Pre-test model selection summary

## Guardrails

- train: 1997–2010,
- validation: 2011–2012,
- common feature set: `{FEATURE_SET_NAME}`,
- test file absent during all pre-test runs,
- full 1997–2014 model dataset absent during all pre-test runs,
- no neural networks,
- model list and hyperparameters frozen before final test evaluation.

## Full-data validation result

The lowest validation MAE was achieved by **{best['model_name']}**
(`{best['model_run_id']}`):

- validation MAE: {float(best['validation_mae']):.6f},
- validation RMSE: {float(best['validation_rmse']):.6f},
- validation R²: {float(best['validation_r2']):.6f}.

Improvement against the mean baseline:

- absolute MAE improvement: {absolute_mean_gain:.6f},
- relative MAE improvement: {relative_mean_gain:.2f} %.

Improvement against the median baseline:

- absolute MAE improvement: {absolute_median_gain:.6f},
- relative MAE improvement: {relative_median_gain:.2f} %.

## Resource-limited validation experiment

KNN and RBF SVR were evaluated separately on a deterministic sample because
their prediction/training costs scale poorly for the full high-dimensional
one-hot encoded dataset.

Resource-limited validation order:

{dataframe_to_markdown(resource_summary[['model_name', 'validation_mae', 'validation_rmse', 'validation_r2']])}

## Frozen decision

All configurations listed in
`data/reference/hw3_frozen_evaluation_plan.json` are frozen for the final
2013–2014 evaluation. Validation metrics may be discussed for model selection.
Final test metrics must not be used for further tuning or changing the model
list.
"""


def freeze_plan() -> None:
    train, validation, manifest = load_pretest_inputs()
    feature_set = list(manifest["feature_sets"][FEATURE_SET_NAME])
    categorical = list(manifest["categorical_features"])
    numeric = [
        column for column in feature_set if column not in categorical
    ]

    full = read_scope_results(FULL_RESULTS_PATH)
    resource = read_scope_results(RESOURCE_RESULTS_PATH)
    require_completed(full, FULL_REQUIRED_RUNS, "full validation")
    require_completed(
        resource, RESOURCE_REQUIRED_RUNS, "resource validation"
    )

    full_completed = full.loc[full["status"].eq("completed")].copy()
    resource_completed = resource.loc[
        resource["status"].eq("completed")
    ].copy()

    full_ordered = full_completed.sort_values(
        ["validation_mae", "validation_rmse", "model_run_id"]
    )
    resource_ordered = resource_completed.sort_values(
        ["validation_mae", "validation_rmse", "model_run_id"]
    )

    full_specs = full_model_specs(categorical, numeric)
    resource_specs = resource_model_specs(categorical, numeric)

    frozen_plan = {
        "task": "HW3 comprehensive regression model evaluation",
        "created_before_final_test": True,
        "configuration_frozen_before_test": True,
        "test_data_present_during_freeze": False,
        "test_data_accessed_by_pretest_script": False,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_tuning": False,
        "train_period": "1997-2010",
        "validation_period": "2011-2012",
        "future_final_test_period": "2013-2014",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "feature_set": FEATURE_SET_NAME,
        "target": TARGET,
        "primary_metric": "MAE",
        "secondary_metrics": ["RMSE", "R2", "MedianAE"],
        "full_validation_models": specs_to_plan(full_specs),
        "resource_limited_validation_models": specs_to_plan(resource_specs),
        "resource_limited_train_rows": int(
            resource_completed["train_rows_used"].max()
        ),
        "resource_limited_validation_rows": int(
            resource_completed["validation_rows_used"].max()
        ),
        "best_full_validation_run_id": str(
            full_ordered.iloc[0]["model_run_id"]
        ),
        "best_resource_validation_run_id": str(
            resource_ordered.iloc[0]["model_run_id"]
        ),
        "full_validation_ranking": [
            {
                "rank": rank,
                "run_id": str(row["model_run_id"]),
                "validation_mae": float(row["validation_mae"]),
                "validation_rmse": float(row["validation_rmse"]),
                "validation_r2": float(row["validation_r2"]),
            }
            for rank, (_, row) in enumerate(
                full_ordered.iterrows(), start=1
            )
        ],
        "resource_validation_ranking": [
            {
                "rank": rank,
                "run_id": str(row["model_run_id"]),
                "validation_mae": float(row["validation_mae"]),
                "validation_rmse": float(row["validation_rmse"]),
                "validation_r2": float(row["validation_r2"]),
            }
            for rank, (_, row) in enumerate(
                resource_ordered.iterrows(), start=1
            )
        ],
        "final_test_rule": (
            "Evaluate every frozen configuration once on 2013-2014. "
            "Do not change model list, preprocessing, features or "
            "hyperparameters after viewing final test metrics."
        ),
        "random_state": RANDOM_STATE,
    }

    FROZEN_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_PLAN_PATH.write_text(
        json.dumps(frozen_plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        build_summary(full_completed, resource_completed),
        encoding="utf-8",
    )

    print(f"full_validation_models={len(full_completed)}")
    print(f"resource_validation_models={len(resource_completed)}")
    print(
        "best_full_validation_run="
        f"{frozen_plan['best_full_validation_run_id']}"
    )
    print(
        "best_resource_validation_run="
        f"{frozen_plan['best_resource_validation_run_id']}"
    )
    print("test_data_present_during_freeze=false")
    print("configuration_frozen_before_test=true")
    print(
        "frozen_plan="
        f"{FROZEN_PLAN_PATH.relative_to(ROOT)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HW3 validation-only model comparison before the final test."
        )
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "full-validation",
            "resource-validation",
            "freeze",
        ],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-run")
    parser.add_argument(
        "--resource-train-rows",
        type=int,
        default=DEFAULT_RESOURCE_TRAIN_ROWS,
    )
    parser.add_argument(
        "--resource-validation-rows",
        type=int,
        default=DEFAULT_RESOURCE_VALIDATION_ROWS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "freeze":
        freeze_plan()
        return 0

    run_phase(
        phase=args.phase,
        resume=args.resume,
        force_run=args.force_run,
        resource_train_rows=args.resource_train_rows,
        resource_validation_rows=args.resource_validation_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
