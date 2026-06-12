from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_hw3_pretest_selection import (
    FEATURE_SET_NAME,
    FULL_REQUIRED_RUNS,
    RANDOM_STATE,
    RESOURCE_REQUIRED_RUNS,
    TARGET,
    deterministic_sample,
    full_model_specs,
    regression_metrics,
    resource_model_specs,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/train_1997_2010.parquet"
VALIDATION_PATH = ROOT / "data/processed/validation_2011_2012.parquet"
TEST_PATH = ROOT / "data/processed/test_2013_2014.parquet"
MANIFEST_PATH = ROOT / "data/reference/model_feature_manifest.json"
FROZEN_PLAN_PATH = ROOT / "data/reference/hw3_frozen_evaluation_plan.json"

PRETEST_FULL_RESULTS_PATH = (
    ROOT / "reports/hw3_pretest_full_validation_results.csv"
)
PRETEST_RESOURCE_RESULTS_PATH = (
    ROOT / "reports/hw3_pretest_resource_validation_results.csv"
)

FULL_RUNS_PATH = ROOT / "data/interim/hw3_final_full_test_runs.csv"
RESOURCE_RUNS_PATH = ROOT / "data/interim/hw3_final_resource_test_runs.csv"
PREDICTION_DIR = ROOT / "data/interim/hw3_final_predictions"

FULL_RESULTS_PATH = ROOT / "reports/hw3_final_full_test_results.csv"
RESOURCE_RESULTS_PATH = ROOT / "reports/hw3_final_resource_test_results.csv"
VALIDATION_TEST_PATH = ROOT / "reports/hw3_validation_test_comparison.csv"
WORST_PREDICTIONS_PATH = ROOT / "reports/hw3_worst_predictions.csv"
RESIDUAL_QUARTILES_PATH = (
    ROOT / "reports/hw3_residual_summary_by_target_quartile.csv"
)
TARGET_DISTRIBUTION_PATH = (
    ROOT / "reports/hw3_target_distribution_comparison.csv"
)
SUMMARY_PATH = ROOT / "reports/hw3_final_evaluation_summary.md"
FINAL_CONFIG_PATH = ROOT / "data/reference/hw3_final_evaluation_record.json"

FULL_SCOPE = "frozen_full_final_test"
RESOURCE_SCOPE = "frozen_resource_limited_final_test"

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
    "test_rows_available",
    "test_rows_used",
    "fit_seconds",
    "train_predict_seconds",
    "test_predict_seconds",
    "train_mae",
    "train_rmse",
    "train_r2",
    "train_median_ae",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_median_ae",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_plan(plan: dict) -> None:
    required_true = [
        "created_before_final_test",
        "configuration_frozen_before_test",
    ]
    for key in required_true:
        if plan.get(key) is not True:
            raise ValueError(f"Frozen plan requires {key}=true.")

    required_false = [
        "test_data_present_during_freeze",
        "test_data_accessed_by_pretest_script",
        "test_used_for_model_selection",
        "test_used_for_hyperparameter_tuning",
    ]
    for key in required_false:
        if plan.get(key) is not False:
            raise ValueError(f"Frozen plan requires {key}=false.")

    if plan.get("train_period") != "1997-2010":
        raise ValueError("Unexpected frozen train period.")
    if plan.get("validation_period") != "2011-2012":
        raise ValueError("Unexpected frozen validation period.")
    if plan.get("future_final_test_period") != "2013-2014":
        raise ValueError("Unexpected frozen test period.")
    if plan.get("feature_set") != FEATURE_SET_NAME:
        raise ValueError("Unexpected frozen feature set.")
    if plan.get("target") != TARGET:
        raise ValueError("Unexpected frozen target.")

    frozen_full_ids = [
        item["run_id"] for item in plan["full_validation_models"]
    ]
    frozen_resource_ids = [
        item["run_id"]
        for item in plan["resource_limited_validation_models"]
    ]
    if frozen_full_ids != FULL_REQUIRED_RUNS:
        raise ValueError(
            "Frozen full model order differs from the pre-registered run list."
        )
    if frozen_resource_ids != RESOURCE_REQUIRED_RUNS:
        raise ValueError(
            "Frozen resource model order differs from the pre-registered run list."
        )


def validate_specs_against_plan(specs: list, frozen_models: list[dict]) -> None:
    by_id = {item["run_id"]: item for item in frozen_models}
    if set(by_id) != {spec.run_id for spec in specs}:
        raise ValueError("Frozen plan and executable specs have different run IDs.")

    for spec in specs:
        frozen = by_id[spec.run_id]
        if spec.model_name != frozen["model_name"]:
            raise ValueError(f"Model name changed for {spec.run_id}.")
        if spec.model_family != frozen["model_family"]:
            raise ValueError(f"Model family changed for {spec.run_id}.")
        if spec.preprocessing_family != frozen["preprocessing_family"]:
            raise ValueError(
                f"Preprocessing family changed for {spec.run_id}."
            )
        if spec.hyperparameters != frozen["hyperparameters"]:
            raise ValueError(f"Hyperparameters changed for {spec.run_id}.")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    required = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        MANIFEST_PATH,
        FROZEN_PLAN_PATH,
        PRETEST_FULL_RESULTS_PATH,
        PRETEST_RESOURCE_RESULTS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final evaluation inputs: {missing}")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with FROZEN_PLAN_PATH.open(encoding="utf-8") as handle:
        plan = json.load(handle)

    validate_frozen_plan(plan)

    train_years = set(int(year) for year in train["Crop_Year"].unique())
    validation_years = set(
        int(year) for year in validation["Crop_Year"].unique()
    )
    test_years = set(int(year) for year in test["Crop_Year"].unique())

    if train_years != set(range(1997, 2011)):
        raise ValueError(f"Unexpected train years: {sorted(train_years)}")
    if validation_years != {2011, 2012}:
        raise ValueError(
            f"Unexpected validation years: {sorted(validation_years)}"
        )
    if test_years != {2013, 2014}:
        raise ValueError(f"Unexpected test years: {sorted(test_years)}")

    if len(train) != int(plan["train_rows"]):
        raise ValueError(
            f"Frozen train row mismatch: {len(train)} != {plan['train_rows']}"
        )
    if len(validation) != int(plan["validation_rows"]):
        raise ValueError(
            "Frozen validation row mismatch: "
            f"{len(validation)} != {plan['validation_rows']}"
        )

    feature_set = list(manifest["feature_sets"][FEATURE_SET_NAME])
    forbidden = set(manifest["forbidden_leakage_columns"])
    leakage = sorted(set(feature_set) & forbidden)
    if leakage:
        raise ValueError(f"Leakage columns in feature set: {leakage}")
    if {"lag_yield_1y", "lag_available"} & set(feature_set):
        raise ValueError("HW3 final comparison must use core_without_lag.")

    for frame_name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if frame[TARGET].isna().any():
            raise ValueError(f"{frame_name} contains missing target values.")

    return train, validation, test, manifest, plan


def empty_result(
    spec,
    scope: str,
    train_rows_available: int,
    train_rows_used: int,
    test_rows_available: int,
    test_rows_used: int,
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
            "fit_period": "1997-2012",
            "evaluation_period": "2013-2014",
            "train_rows_available": train_rows_available,
            "train_rows_used": train_rows_used,
            "test_rows_available": test_rows_available,
            "test_rows_used": test_rows_used,
            "status": "pending",
            "warning_count": 0,
            "warning_summary": "",
            "error_type": "",
            "error_message": "",
            "random_state": RANDOM_STATE,
        }
    )
    return row


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[RESULT_COLUMNS]


def write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = read_results(path)
    frame = frame.loc[
        ~frame["model_run_id"].astype(str).eq(str(row["model_run_id"]))
    ]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame[RESULT_COLUMNS].sort_values("model_run_id")
    frame.to_csv(path, index=False)


def completed_ids(path: Path) -> set[str]:
    frame = read_results(path)
    if frame.empty:
        return set()
    return set(
        frame.loc[frame["status"].eq("completed"), "model_run_id"].astype(str)
    )


def prediction_path(scope: str, run_id: str) -> Path:
    safe_scope = "full" if scope == FULL_SCOPE else "resource"
    return PREDICTION_DIR / f"{safe_scope}_{run_id}.npy"


def run_one(
    spec,
    scope: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_set: list[str],
    train_rows_available: int,
    test_rows_available: int,
) -> dict:
    row = empty_result(
        spec,
        scope,
        train_rows_available,
        len(train_frame),
        test_rows_available,
        len(test_frame),
    )
    estimator = spec.build_estimator()
    X_train = train_frame[feature_set]
    y_train = train_frame[TARGET]
    X_test = test_frame[feature_set]
    y_test = test_frame[TARGET]

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
                test_prediction = estimator.predict(
                    np.zeros((len(test_frame), 1))
                )
            else:
                test_prediction = estimator.predict(X_test)
            test_predict_seconds = time.perf_counter() - started

            warning_messages = [
                f"{item.category.__name__}: {item.message}" for item in caught
            ]

        train_metrics = regression_metrics(y_train, train_prediction)
        test_metrics = regression_metrics(y_test, test_prediction)

        row.update(
            {
                "fit_seconds": fit_seconds,
                "train_predict_seconds": train_predict_seconds,
                "test_predict_seconds": test_predict_seconds,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_median_ae": train_metrics["median_ae"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_median_ae": test_metrics["median_ae"],
                "mae_generalization_gap": (
                    test_metrics["mae"] - train_metrics["mae"]
                ),
                "rmse_generalization_gap": (
                    test_metrics["rmse"] - train_metrics["rmse"]
                ),
                "r2_generalization_gap": (
                    test_metrics["r2"] - train_metrics["r2"]
                ),
                "status": "completed",
                "warning_count": len(warning_messages),
                "warning_summary": " | ".join(warning_messages)[:4000],
            }
        )

        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
        np.save(
            prediction_path(scope, spec.run_id),
            np.asarray(test_prediction, dtype=float),
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
) -> None:
    train, validation, test, manifest, plan = load_inputs()
    final_train = pd.concat([train, validation], ignore_index=True)

    feature_set = list(manifest["feature_sets"][FEATURE_SET_NAME])
    categorical = list(manifest["categorical_features"])
    numeric = [
        column for column in feature_set if column not in categorical
    ]

    if phase == "full-test":
        specs = full_model_specs(categorical, numeric)
        validate_specs_against_plan(
            specs, plan["full_validation_models"]
        )
        train_frame = final_train
        test_frame = test.reset_index(drop=True)
        results_path = FULL_RUNS_PATH
        scope = FULL_SCOPE
    elif phase == "resource-test":
        specs = resource_model_specs(categorical, numeric)
        validate_specs_against_plan(
            specs, plan["resource_limited_validation_models"]
        )
        resource_train_rows = int(plan["resource_limited_train_rows"])
        resource_test_rows = int(plan["resource_limited_validation_rows"])
        train_frame = deterministic_sample(
            final_train, resource_train_rows, RANDOM_STATE
        ).reset_index(drop=True)
        test_frame = deterministic_sample(
            test, resource_test_rows, RANDOM_STATE
        ).reset_index(drop=True)
        results_path = RESOURCE_RUNS_PATH
        scope = RESOURCE_SCOPE
    else:
        raise ValueError(f"Unsupported test phase: {phase}")

    if force_run:
        specs = [spec for spec in specs if spec.run_id == force_run]
        if not specs:
            raise ValueError(f"Unknown run ID for {phase}: {force_run}")

    completed = completed_ids(results_path)
    for spec in specs:
        pred_path = prediction_path(scope, spec.run_id)
        if (
            resume
            and spec.run_id in completed
            and pred_path.exists()
            and force_run is None
        ):
            print(f"model_run_id={spec.run_id} status=skipped_resume")
            continue

        row = run_one(
            spec=spec,
            scope=scope,
            train_frame=train_frame,
            test_frame=test_frame,
            feature_set=feature_set,
            train_rows_available=len(final_train),
            test_rows_available=len(test),
        )
        write_result(results_path, row)
        print(
            f"model_run_id={spec.run_id} "
            f"status={row['status']} "
            f"test_mae={row.get('test_mae')}"
        )


def require_completed(
    frame: pd.DataFrame,
    required_ids: list[str],
    label: str,
) -> pd.DataFrame:
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed_ids_set = set(completed["model_run_id"].astype(str))
    missing = [
        run_id for run_id in required_ids if run_id not in completed_ids_set
    ]
    if missing:
        raise RuntimeError(
            f"Cannot finalize {label}. Missing completed runs: "
            + ", ".join(missing)
        )
    return completed.loc[
        completed["model_run_id"].astype(str).isin(required_ids)
    ].copy()


def target_distribution_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    final_train = pd.concat([train, validation], ignore_index=True)
    rows = []
    for label, frame in [
        ("train_1997_2010", train),
        ("validation_2011_2012", validation),
        ("final_train_1997_2012", final_train),
        ("test_2013_2014", test),
    ]:
        y = frame[TARGET].astype(float)
        rows.append(
            {
                "split": label,
                "rows": len(frame),
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "p50": float(y.quantile(0.50)),
                "p90": float(y.quantile(0.90)),
                "p95": float(y.quantile(0.95)),
                "p99": float(y.quantile(0.99)),
                "max": float(y.max()),
            }
        )
    return pd.DataFrame(rows)


def residual_quartile_summary(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "prediction": np.asarray(prediction, dtype=float),
        }
    )
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["absolute_error"] = frame["residual"].abs()
    frame["target_quartile"] = pd.qcut(
        frame["actual"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )

    rows = []
    for quartile, group in frame.groupby(
        "target_quartile", observed=True, sort=True
    ):
        rows.append(
            {
                "target_quartile": str(quartile),
                "rows": len(group),
                "actual_min": float(group["actual"].min()),
                "actual_max": float(group["actual"].max()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(
                    np.sqrt(np.mean(np.square(group["residual"])))
                ),
                "mean_residual_actual_minus_prediction": float(
                    group["residual"].mean()
                ),
                "underprediction_rate_pct": float(
                    (group["residual"] > 0).mean() * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    full_results: pd.DataFrame,
    comparison: pd.DataFrame,
    final_train: pd.DataFrame,
    test: pd.DataFrame,
    best_prediction: np.ndarray,
) -> None:
    plot_specs = [
        ("test_mae", "Test MAE podle modelu", "MAE", "hw3_final_test_mae.png"),
        (
            "test_rmse",
            "Test RMSE podle modelu",
            "RMSE",
            "hw3_final_test_rmse.png",
        ),
        ("test_r2", "Test R² podle modelu", "R²", "hw3_final_test_r2.png"),
    ]
    for column, title, ylabel, filename in plot_specs:
        ordered = full_results.sort_values(
            column, ascending=(column != "test_r2")
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(ordered["model_name"], ordered[column])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    ordered = comparison.sort_values("validation_mae")
    x = np.arange(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, ordered["validation_mae"], width, label="validation")
    ax.bar(x + width / 2, ordered["test_mae"], width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model_name"], rotation=40, ha="right")
    ax.set_title("Validation versus test MAE")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        ROOT / "reports/hw3_validation_vs_test_mae.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    actual = test[TARGET].to_numpy(dtype=float)
    prediction = np.asarray(best_prediction, dtype=float)
    residual = actual - prediction

    sample_positions = np.linspace(
        0, len(test) - 1, num=min(5000, len(test)), dtype=int
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        actual[sample_positions],
        prediction[sample_positions],
        s=8,
        alpha=0.35,
    )
    lower = float(min(actual.min(), prediction.min()))
    upper = float(max(actual.max(), prediction.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title("Random Forest: skutečnost versus predikce")
    ax.set_xlabel("Skutečný výnos")
    ax.set_ylabel("Predikovaný výnos")
    fig.tight_layout()
    fig.savefig(
        ROOT / "reports/hw3_best_actual_vs_predicted.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        prediction[sample_positions],
        residual[sample_positions],
        s=8,
        alpha=0.35,
    )
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Random Forest: rezidua versus predikce")
    ax.set_xlabel("Predikovaný výnos")
    ax.set_ylabel("Reziduum (skutečnost − predikce)")
    fig.tight_layout()
    fig.savefig(
        ROOT / "reports/hw3_best_residuals_vs_predicted.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    clip = float(np.quantile(np.abs(residual), 0.995))
    clipped_residual = residual[np.abs(residual) <= clip]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clipped_residual, bins=60)
    ax.set_title("Random Forest: rozdělení reziduí (do 99,5. percentilu)")
    ax.set_xlabel("Reziduum")
    ax.set_ylabel("Počet řádků")
    fig.tight_layout()
    fig.savefig(
        ROOT / "reports/hw3_best_residual_histogram.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    cutoff = float(
        max(
            final_train[TARGET].quantile(0.99),
            test[TARGET].quantile(0.99),
        )
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        final_train.loc[final_train[TARGET] <= cutoff, TARGET],
        bins=60,
        density=True,
        alpha=0.55,
        label="final train 1997–2012",
    )
    ax.hist(
        test.loc[test[TARGET] <= cutoff, TARGET],
        bins=60,
        density=True,
        alpha=0.55,
        label="test 2013–2014",
    )
    ax.set_title("Rozdělení targetu do společného 99. percentilu")
    ax.set_xlabel("target_yield")
    ax.set_ylabel("Hustota")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        ROOT / "reports/hw3_target_distribution.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )

    headers = [str(column) for column in display.columns]
    rows = [headers] + display.astype(str).values.tolist()
    widths = [
        max(len(row[index]) for row in rows)
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )

    separator = [
        "-" * max(3, widths[index]) for index in range(len(headers))
    ]
    return "\n".join(
        [render(headers), render(separator)]
        + [render(row) for row in rows[1:]]
    )


def build_summary(
    full: pd.DataFrame,
    resource: pd.DataFrame,
    comparison: pd.DataFrame,
    plan: dict,
    residual_quartiles: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> str:
    selected_run = str(plan["best_full_validation_run_id"])
    selected = full.loc[full["model_run_id"].eq(selected_run)].iloc[0]
    test_best = full.sort_values("test_mae").iloc[0]
    mean_baseline = full.loc[full["model_run_id"].eq("dummy_mean")].iloc[0]
    median_baseline = full.loc[full["model_run_id"].eq("dummy_median")].iloc[0]
    validation_selected = comparison.loc[
        comparison["model_run_id"].eq(selected_run)
    ].iloc[0]

    improvement_mean = (
        float(mean_baseline["test_mae"]) - float(selected["test_mae"])
    )
    improvement_median = (
        float(median_baseline["test_mae"]) - float(selected["test_mae"])
    )

    return f"""# HW3 – Final frozen test evaluation

## Experimental protocol

- model list, features, preprocessing and hyperparameters were frozen before
  the final test file was restored,
- model-selection period: train 1997–2010 and validation 2011–2012,
- final fit period: 1997–2012,
- final test period: 2013–2014,
- common feature set: `core_without_lag`,
- primary metric: MAE,
- final test metrics were not used for further tuning.

Frozen-plan SHA-256:

`{sha256_file(FROZEN_PLAN_PATH)}`

## Pre-selected model

Validation selected **{selected['model_name']}**
(`{selected_run}`).

Validation metrics:

- MAE: {float(validation_selected['validation_mae']):.6f},
- RMSE: {float(validation_selected['validation_rmse']):.6f},
- R²: {float(validation_selected['validation_r2']):.6f}.

Final test metrics:

- MAE: {float(selected['test_mae']):.6f},
- RMSE: {float(selected['test_rmse']):.6f},
- R²: {float(selected['test_r2']):.6f},
- MedianAE: {float(selected['test_median_ae']):.6f}.

Improvement against the mean baseline:

- absolute MAE: {improvement_mean:.6f},
- relative MAE: {improvement_mean / float(mean_baseline['test_mae']) * 100.0:.2f} %.

Improvement against the median baseline:

- absolute MAE: {improvement_median:.6f},
- relative MAE: {improvement_median / float(median_baseline['test_mae']) * 100.0:.2f} %.

## Test ranking

The test ranking is reported for comparison only. It does not change the
pre-selected model or any configuration.

Test-best run: **{test_best['model_name']}**
(`{test_best['model_run_id']}`), test MAE
{float(test_best['test_mae']):.6f}.

{dataframe_to_markdown(full[['model_name', 'test_mae', 'test_rmse', 'test_r2', 'mae_generalization_gap']].sort_values('test_mae'))}

## Validation-to-test stability

{dataframe_to_markdown(comparison[['model_name', 'validation_mae', 'test_mae', 'test_minus_validation_mae', 'validation_rmse', 'test_rmse', 'validation_r2', 'test_r2']].sort_values('validation_mae'))}

A train–test or validation–test gap is not automatically proof of
overfitting. Because the split is chronological, the difference can also
reflect temporal distribution shift between historical and later years.

## Residual analysis of the validation-selected Random Forest

Positive residual means underprediction; negative residual means
overprediction.

{dataframe_to_markdown(residual_quartiles)}

## Target distribution comparison

{dataframe_to_markdown(target_summary)}

## Resource-limited experiment

KNN and RBF-SVR remain in a separate deterministic sample experiment because
their costs scale poorly on the full high-dimensional dataset. Their metrics
must not be mixed directly with the full-data ranking.

{dataframe_to_markdown(resource[['model_name', 'test_mae', 'test_rmse', 'test_r2']].sort_values('test_mae'))}

## Interpretation limits

The models predict crop yield, not economic profit and not a causal effect.
The dataset does not include complete costs of labour, irrigation,
fertiliser, pesticides, transport or crop selling prices.
"""


def finalize() -> None:
    train, validation, test, _, plan = load_inputs()
    final_train = pd.concat([train, validation], ignore_index=True)

    full = require_completed(
        read_results(FULL_RUNS_PATH), FULL_REQUIRED_RUNS, "full test"
    )
    resource = require_completed(
        read_results(RESOURCE_RUNS_PATH),
        RESOURCE_REQUIRED_RUNS,
        "resource test",
    )

    pretest_full = pd.read_csv(PRETEST_FULL_RESULTS_PATH)
    pretest_full = pretest_full.loc[
        pretest_full["status"].eq("completed"),
        [
            "model_run_id",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "validation_median_ae",
        ],
    ].copy()

    comparison = pretest_full.merge(
        full[
            [
                "model_run_id",
                "model_name",
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_median_ae",
            ]
        ],
        on="model_run_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(FULL_REQUIRED_RUNS):
        raise RuntimeError(
            "Validation/test comparison does not contain all frozen full runs."
        )
    comparison["test_minus_validation_mae"] = (
        comparison["test_mae"] - comparison["validation_mae"]
    )
    comparison["test_minus_validation_rmse"] = (
        comparison["test_rmse"] - comparison["validation_rmse"]
    )
    comparison["test_minus_validation_r2"] = (
        comparison["test_r2"] - comparison["validation_r2"]
    )

    selected_run = str(plan["best_full_validation_run_id"])
    selected_prediction = np.load(
        prediction_path(FULL_SCOPE, selected_run)
    )
    if len(selected_prediction) != len(test):
        raise RuntimeError("Selected prediction length does not match test.")

    test_reset = test.reset_index(drop=True)
    actual = test_reset[TARGET].to_numpy(dtype=float)
    residual = actual - selected_prediction
    absolute_error = np.abs(residual)

    identity_columns = [
        column
        for column in [
            "canonical_crop_row_id",
            "Crop_Year",
            "canonical_state_name",
            "canonical_district_name",
            "Crop_canonical",
            "Season_canonical",
            "Area_corrected",
            TARGET,
        ]
        if column in test_reset.columns
    ]
    worst_positions = np.argsort(-absolute_error)[:50]
    worst = test_reset.iloc[worst_positions][identity_columns].copy()
    worst["prediction"] = selected_prediction[worst_positions]
    worst["residual_actual_minus_prediction"] = residual[worst_positions]
    worst["absolute_error"] = absolute_error[worst_positions]
    worst = worst.sort_values("absolute_error", ascending=False)

    residual_quartiles = residual_quartile_summary(
        test_reset[TARGET], selected_prediction
    )
    target_summary = target_distribution_summary(
        train, validation, test_reset
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    full.sort_values("test_mae").to_csv(FULL_RESULTS_PATH, index=False)
    resource.sort_values("test_mae").to_csv(
        RESOURCE_RESULTS_PATH, index=False
    )
    comparison.sort_values("validation_mae").to_csv(
        VALIDATION_TEST_PATH, index=False
    )
    worst.to_csv(WORST_PREDICTIONS_PATH, index=False)
    residual_quartiles.to_csv(RESIDUAL_QUARTILES_PATH, index=False)
    target_summary.to_csv(TARGET_DISTRIBUTION_PATH, index=False)

    save_plots(
        full_results=full,
        comparison=comparison,
        final_train=final_train,
        test=test_reset,
        best_prediction=selected_prediction,
    )

    SUMMARY_PATH.write_text(
        build_summary(
            full,
            resource,
            comparison,
            plan,
            residual_quartiles,
            target_summary,
        ),
        encoding="utf-8",
    )

    record = {
        "task": "HW3 final frozen regression model evaluation",
        "frozen_plan_path": str(FROZEN_PLAN_PATH.relative_to(ROOT)),
        "frozen_plan_sha256": sha256_file(FROZEN_PLAN_PATH),
        "configuration_frozen_before_test": True,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_tuning": False,
        "post_test_tuning_performed": False,
        "final_fit_period": "1997-2012",
        "final_test_period": "2013-2014",
        "final_train_rows": len(final_train),
        "test_rows": len(test_reset),
        "selected_from_validation_run_id": selected_run,
        "reported_test_best_run_id": str(
            full.sort_values("test_mae").iloc[0]["model_run_id"]
        ),
        "full_model_count": len(full),
        "resource_model_count": len(resource),
        "random_state": RANDOM_STATE,
    }
    FINAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_CONFIG_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"full_test_models={len(full)}")
    print(f"resource_test_models={len(resource)}")
    print(f"selected_from_validation={selected_run}")
    print(
        "reported_test_best="
        f"{full.sort_values('test_mae').iloc[0]['model_run_id']}"
    )
    print("post_test_tuning_performed=false")
    print(f"summary={SUMMARY_PATH.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen HW3 model plan on the final test."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["full-test", "resource-test", "finalize"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "finalize":
        finalize()
        return 0
    run_phase(args.phase, args.resume, args.force_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
