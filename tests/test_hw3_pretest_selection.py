from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "run_hw3_pretest_selection.py"
)
SPEC = importlib.util.spec_from_file_location("hw3_pretest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
hw3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hw3
SPEC.loader.exec_module(hw3)


def test_regression_metrics() -> None:
    metrics = hw3.regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 4.0]),
    )
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["median_ae"] == pytest.approx(0.0)


def test_deterministic_sample_is_reproducible() -> None:
    frame = pd.DataFrame({"x": range(100)})
    first = hw3.deterministic_sample(frame, 10, 42)
    second = hw3.deterministic_sample(frame, 10, 42)
    pd.testing.assert_frame_equal(first, second)


def test_deterministic_sample_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        hw3.deterministic_sample(pd.DataFrame({"x": [1]}), 0)


def test_full_model_families_include_seminar_models() -> None:
    specs = hw3.full_model_specs(["category"], ["numeric"])
    run_ids = {spec.run_id for spec in specs}
    assert "dummy_mean" in run_ids
    assert "linear_regression" in run_ids
    assert "ridge_alpha_1" in run_ids
    assert "lasso_alpha_0_01" in run_ids
    assert "elastic_net_alpha_0_01_l1_0_5" in run_ids
    assert "decision_tree_leaf_20" in run_ids
    assert "random_forest_200_leaf_20" in run_ids
    assert "gradient_boosting_60_lr_0_1_depth_3" in run_ids
    assert "linear_svr_c_0_03" in run_ids


def test_resource_models_include_knn_and_rbf_svr() -> None:
    specs = hw3.resource_model_specs(["category"], ["numeric"])
    run_ids = {spec.run_id for spec in specs}
    assert "sample_knn_15_distance" in run_ids
    assert "sample_svr_rbf_c_10_epsilon_5" in run_ids


def test_pretest_paths_do_not_point_to_test_results() -> None:
    result_paths = {
        hw3.FULL_RESULTS_PATH.name,
        hw3.RESOURCE_RESULTS_PATH.name,
        hw3.SUMMARY_PATH.name,
        hw3.FROZEN_PLAN_PATH.name,
    }
    assert all("test_result" not in name for name in result_paths)


def test_holdout_guard_rejects_present_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "test.parquet"
    test_path.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(hw3, "HOLDOUT_TEST_PATH", test_path)
    monkeypatch.setattr(
        hw3,
        "FULL_MODEL_DATASET_PATH",
        tmp_path / "missing.parquet",
    )
    with pytest.raises(RuntimeError, match="refuses to run"):
        hw3.assert_holdout_absent()


def test_holdout_guard_accepts_absent_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        hw3, "HOLDOUT_TEST_PATH", tmp_path / "missing-test.parquet"
    )
    monkeypatch.setattr(
        hw3,
        "FULL_MODEL_DATASET_PATH",
        tmp_path / "missing-full.parquet",
    )
    hw3.assert_holdout_absent()


def test_result_schema_contains_validation_metrics() -> None:
    required = {
        "train_mae",
        "validation_mae",
        "train_rmse",
        "validation_rmse",
        "train_r2",
        "validation_r2",
        "mae_generalization_gap",
    }
    assert required.issubset(set(hw3.RESULT_COLUMNS))


def test_required_run_ids_are_unique() -> None:
    combined = hw3.FULL_REQUIRED_RUNS + hw3.RESOURCE_REQUIRED_RUNS
    assert len(combined) == len(set(combined))



def test_summary_contains_all_resource_models() -> None:
    full = pd.DataFrame(
        [
            {
                "model_run_id": "dummy_mean",
                "model_name": "DummyRegressor (mean)",
                "validation_mae": 5.0,
                "validation_rmse": 6.0,
                "validation_r2": 0.0,
            },
            {
                "model_run_id": "dummy_median",
                "model_name": "DummyRegressor (median)",
                "validation_mae": 4.0,
                "validation_rmse": 6.5,
                "validation_r2": -0.1,
            },
            {
                "model_run_id": "random_forest_200_leaf_20",
                "model_name": "Random Forest",
                "validation_mae": 1.5,
                "validation_rmse": 3.0,
                "validation_r2": 0.8,
            },
        ]
    )

    resource_rows = []
    for index, run_id in enumerate(hw3.RESOURCE_REQUIRED_RUNS):
        resource_rows.append(
            {
                "model_run_id": run_id,
                "model_name": f"resource-{run_id}",
                "validation_mae": 1.0 + index,
                "validation_rmse": 2.0 + index,
                "validation_r2": 0.8 - index * 0.1,
            }
        )
    resource = pd.DataFrame(resource_rows)

    summary = hw3.build_summary(full, resource)

    for run_id in hw3.RESOURCE_REQUIRED_RUNS:
        assert f"resource-{run_id}" in summary

    assert (
        "their prediction/training costs scale poorly for the full "
        "high-dimensional"
    ) in summary
