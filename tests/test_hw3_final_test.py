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
    / "run_hw3_final_test.py"
)
SPEC = importlib.util.spec_from_file_location("hw3_final", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
hw3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hw3
SPEC.loader.exec_module(hw3)


def valid_plan() -> dict:
    return {
        "created_before_final_test": True,
        "configuration_frozen_before_test": True,
        "test_data_present_during_freeze": False,
        "test_data_accessed_by_pretest_script": False,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_tuning": False,
        "train_period": "1997-2010",
        "validation_period": "2011-2012",
        "future_final_test_period": "2013-2014",
        "feature_set": hw3.FEATURE_SET_NAME,
        "target": hw3.TARGET,
        "full_validation_models": [
            {"run_id": run_id} for run_id in hw3.FULL_REQUIRED_RUNS
        ],
        "resource_limited_validation_models": [
            {"run_id": run_id} for run_id in hw3.RESOURCE_REQUIRED_RUNS
        ],
    }


def test_validate_frozen_plan_accepts_expected_flags() -> None:
    hw3.validate_frozen_plan(valid_plan())


@pytest.mark.parametrize(
    "key",
    [
        "created_before_final_test",
        "configuration_frozen_before_test",
    ],
)
def test_validate_frozen_plan_rejects_required_true_flag(key: str) -> None:
    plan = valid_plan()
    plan[key] = False
    with pytest.raises(ValueError):
        hw3.validate_frozen_plan(plan)


@pytest.mark.parametrize(
    "key",
    [
        "test_data_present_during_freeze",
        "test_data_accessed_by_pretest_script",
        "test_used_for_model_selection",
        "test_used_for_hyperparameter_tuning",
    ],
)
def test_validate_frozen_plan_rejects_required_false_flag(
    key: str,
) -> None:
    plan = valid_plan()
    plan[key] = True
    with pytest.raises(ValueError):
        hw3.validate_frozen_plan(plan)


def test_residual_quartile_summary_uses_actual_minus_prediction() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([0.0, 2.0, 4.0, 5.0])
    result = hw3.residual_quartile_summary(actual, prediction)
    assert result["rows"].sum() == 4
    assert result["underprediction_rate_pct"].between(0, 100).all()


def test_target_distribution_summary_contains_all_splits() -> None:
    train = pd.DataFrame({hw3.TARGET: [1.0, 2.0]})
    validation = pd.DataFrame({hw3.TARGET: [3.0]})
    test = pd.DataFrame({hw3.TARGET: [4.0, 5.0]})
    result = hw3.target_distribution_summary(train, validation, test)
    assert result["split"].tolist() == [
        "train_1997_2010",
        "validation_2011_2012",
        "final_train_1997_2012",
        "test_2013_2014",
    ]
    assert result.loc[
        result["split"].eq("final_train_1997_2012"), "rows"
    ].iloc[0] == 3


def test_dataframe_to_markdown_does_not_require_tabulate() -> None:
    frame = pd.DataFrame({"name": ["A"], "metric": [1.23456789]})
    rendered = hw3.dataframe_to_markdown(frame)
    assert "| name" in rendered
    assert "1.234568" in rendered


def test_result_columns_include_train_test_metrics() -> None:
    expected = {
        "train_mae",
        "test_mae",
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
        "mae_generalization_gap",
    }
    assert expected.issubset(set(hw3.RESULT_COLUMNS))


def test_prediction_paths_separate_scopes() -> None:
    full = hw3.prediction_path(hw3.FULL_SCOPE, "model")
    resource = hw3.prediction_path(hw3.RESOURCE_SCOPE, "model")
    assert full != resource
    assert full.name.startswith("full_")
    assert resource.name.startswith("resource_")
