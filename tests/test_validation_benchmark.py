from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_validation_benchmark as benchmark  # noqa: E402


def manifest() -> dict[str, object]:
    return {
        "target_column": "target_yield",
        "categorical_features": ["Crop_canonical"],
        "feature_sets": {
            "core_without_lag": ["Crop_canonical", "Crop_Year", "rain_sum_mm"],
            "core_with_lag": ["Crop_canonical", "Crop_Year", "rain_sum_mm", "lag_yield_1y", "lag_available"],
        },
        "forbidden_leakage_columns": ["target_yield", "Production_corrected"],
    }


def rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "canonical_crop_row_id": ["t1", "t2", "t3", "t4"],
            "Crop_Year": [2009, 2010, 2010, 2010],
            "Crop_canonical": ["Rice", "Rice", "Wheat", "Wheat"],
            "canonical_state_name": ["S", "S", "S", "S"],
            "canonical_district_name": ["D1", "D1", "D2", "D2"],
            "Season_canonical": ["Kharif", "Kharif", "Rabi", "Rabi"],
            "rain_sum_mm": [1.0, 2.0, 3.0, 4.0],
            "lag_yield_1y": [np.nan, 1.0, np.nan, 10.0],
            "lag_available": [0, 1, 0, 1],
            "target_yield": [1.0, 3.0, 10.0, 14.0],
        }
    )
    validation = pd.DataFrame(
        {
            "canonical_crop_row_id": ["v1", "v2", "v3"],
            "Crop_Year": [2011, 2011, 2012],
            "Crop_canonical": ["Rice", "Wheat", "NewCrop"],
            "canonical_state_name": ["S", "S", "S"],
            "canonical_district_name": ["D1", "D2", "D3"],
            "Season_canonical": ["Kharif", "Rabi", "Kharif"],
            "rain_sum_mm": [5.0, 6.0, 7.0],
            "lag_yield_1y": [3.0, np.nan, 99.0],
            "lag_available": [1, 0, 1],
            "target_yield": [4.0, 12.0, 9.0],
        }
    )
    return train, validation


def completed_result(model_run_id: str, phase: str = "models", mae: float = 1.0, rmse: float = 2.0, r2: float = 0.5) -> dict[str, object]:
    result = {column: "" for column in benchmark.RUN_COLUMNS}
    result.update(
        {
            "model_run_id": model_run_id,
            "phase": phase,
            "model_name": "Ridge",
            "model_family": "linear",
            "feature_set": "core_without_lag",
            "preprocessing_family": "linear",
            "hyperparameters_json": "{}",
            "model_parameters": "{}",
            "training_scope": "full_train",
            "train_rows_available": 4,
            "train_rows_used": 4,
            "validation_rows": 3,
            "fit_seconds": 0.1,
            "predict_seconds": 0.1,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "median_ae": mae,
            "lag_subset_mae": mae,
            "lag_subset_rows": 2,
            "status": "completed",
            "warning_count": 0,
            "warning_summary": "",
            "error_type": "",
            "error_message": "",
            "random_state": 42,
            "test_data_accessed": False,
        }
    )
    return result


def test_global_median_fits_only_on_train() -> None:
    train, validation = rows()
    config = benchmark.baseline_configs()[0]
    result = benchmark.execute_model_config(config, train, validation, manifest(), write_prediction_file=False)
    predictions = np.repeat(np.median(train["target_yield"]), len(validation))
    expected = benchmark.compute_regression_metrics(validation["target_yield"], predictions)
    assert result["mae"] == expected["mae"]


def test_crop_median_uses_train_mapping() -> None:
    train, validation = rows()
    model = benchmark.CropMedianRegressor().fit(train[["Crop_canonical"]], train["target_yield"])
    predictions = model.predict(validation[["Crop_canonical"]])
    assert predictions[0] == 2.0
    assert predictions[1] == 12.0


def test_unseen_crop_uses_global_train_median() -> None:
    train, validation = rows()
    model = benchmark.CropMedianRegressor().fit(train[["Crop_canonical"]], train["target_yield"])
    prediction = model.predict(validation.loc[[2], ["Crop_canonical"]])[0]
    assert prediction == np.median(train["target_yield"])


def test_lag_baseline_uses_lag_when_available() -> None:
    train, validation = rows()
    model = benchmark.LagWithCropMedianFallbackRegressor().fit(train, train["target_yield"])
    predictions = model.predict(validation)
    assert predictions[0] == 3.0


def test_lag_baseline_uses_fallback_when_lag_missing() -> None:
    train, validation = rows()
    model = benchmark.LagWithCropMedianFallbackRegressor().fit(train, train["target_yield"])
    predictions = model.predict(validation)
    assert predictions[1] == 12.0


def test_mae_is_computed_correctly() -> None:
    metrics = benchmark.compute_regression_metrics([1.0, 3.0], [2.0, 1.0])
    assert metrics["mae"] == 1.5


def test_rmse_is_computed_correctly() -> None:
    metrics = benchmark.compute_regression_metrics([1.0, 3.0], [2.0, 1.0])
    assert metrics["rmse"] == math.sqrt((1.0 + 4.0) / 2.0)


def test_r2_is_computed_correctly() -> None:
    metrics = benchmark.compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["r2"] == 1.0


def test_target_is_not_allowed_in_features() -> None:
    train, _ = rows()
    with pytest.raises(ValueError, match="target_yield"):
        benchmark.validate_feature_set("bad", ["target_yield"], train.columns, manifest())


def test_forbidden_leakage_column_raises() -> None:
    train, _ = rows()
    with pytest.raises(ValueError, match="forbidden leakage"):
        benchmark.validate_feature_set("bad", ["Crop_canonical", "Production_corrected"], train.columns, manifest())


def test_one_hot_encoder_uses_handle_unknown_ignore() -> None:
    train, validation = rows()
    features = ["Crop_canonical", "Crop_Year", "rain_sum_mm"]
    preprocessor = benchmark.make_linear_preprocessor(features, manifest())
    preprocessor.fit(train[features])
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    assert encoder.handle_unknown == "ignore"
    preprocessor.transform(validation[features])


def test_unknown_ordinal_category_gets_minus_one() -> None:
    train, validation = rows()
    features = ["Crop_canonical", "Crop_Year", "rain_sum_mm"]
    preprocessor = benchmark.make_tree_preprocessor(features, manifest())
    preprocessor.fit(train[features])
    transformed = np.asarray(preprocessor.transform(validation.loc[[2], features]))
    assert transformed[0, 0] == -1


def test_preprocessing_is_fit_only_on_train() -> None:
    train, validation = rows()
    features = ["Crop_canonical", "Crop_Year", "rain_sum_mm"]
    preprocessor = benchmark.make_linear_preprocessor(features, manifest())
    preprocessor.fit(train[features])
    preprocessor.transform(validation[features])
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    assert "NewCrop" not in set(encoder.categories_[0])


def test_test_dataset_is_not_opened() -> None:
    with pytest.raises(ValueError, match="test dataset"):
        benchmark.guard_not_test_path(benchmark.TEST_PATH)
    assert benchmark.TEST_DATA_ACCESSED is False


def test_resume_skips_completed_run() -> None:
    config = benchmark.baseline_configs()[0]
    existing = pd.DataFrame([completed_result(config.model_run_id, phase="baselines")])
    assert benchmark.should_skip_run(config, existing, resume=True)


def test_failed_run_does_not_stop_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    train, validation = rows()
    configs = [
        benchmark.BenchmarkConfig("bad", "models", "bad", "BadModel", "linear", "core_without_lag", "linear", {}),
        benchmark.BenchmarkConfig("good", "models", "good", "Ridge", "linear", "core_without_lag", "linear", {}),
    ]
    calls: list[str] = []

    def fake_read_runs(path: Path = benchmark.RUNS_PATH) -> pd.DataFrame:
        return pd.DataFrame(columns=benchmark.RUN_COLUMNS)

    def evaluator(config, train_frame, validation_frame, manifest_dict, seed):
        calls.append(config.model_run_id)
        if config.model_run_id == "bad":
            raise RuntimeError("planned failure")
        return completed_result(config.model_run_id)

    written: list[dict[str, object]] = []
    monkeypatch.setattr(benchmark, "read_runs", fake_read_runs)
    benchmark.run_config_sequence(
        configs,
        train,
        validation,
        manifest(),
        resume=False,
        force_model=None,
        random_state=42,
        evaluator=evaluator,
        writer=written.append,
    )
    assert calls == ["bad", "good"]
    assert [item["status"] for item in written] == ["failed", "completed"]


def test_feature_set_selection_uses_validation_metrics() -> None:
    comparison = pd.DataFrame(
        [
            completed_result("ridge_without", "feature-sets", mae=1.0, rmse=10.0, r2=0.0)
            | {"model_name": "Ridge", "feature_set": "core_without_lag"},
            completed_result("ridge_with", "feature-sets", mae=2.0, rmse=1.0, r2=0.9)
            | {"model_name": "Ridge", "feature_set": "core_with_lag"},
            completed_result("tree_without", "feature-sets", mae=1.1, rmse=10.0, r2=0.0)
            | {"model_name": "DecisionTree", "feature_set": "core_without_lag"},
            completed_result("tree_with", "feature-sets", mae=2.1, rmse=1.0, r2=0.9)
            | {"model_name": "DecisionTree", "feature_set": "core_with_lag"},
        ]
    )
    selected, _ = benchmark.select_feature_set(comparison)
    assert selected == "core_without_lag"


def test_primary_selection_metric_is_mae() -> None:
    results = pd.DataFrame(
        [
            completed_result("low_mae", "models", mae=1.0, rmse=100.0, r2=-1.0)
            | {"model_name": "Ridge"},
            completed_result("low_rmse", "models", mae=2.0, rmse=1.0, r2=0.99)
            | {"model_name": "LinearRegression"},
        ]
    )
    winner = benchmark.select_winning_model(results)
    assert winner["model_run_id"] == "low_mae"


def test_frozen_config_contains_test_data_accessed_false() -> None:
    selected = pd.Series(completed_result("winner"))
    frozen = benchmark.build_frozen_configuration(selected, pd.DataFrame([completed_result("baseline", "baselines")]), 42)
    assert frozen["test_data_accessed"] is False


def test_knn_training_scope_is_recorded() -> None:
    configs = benchmark.real_model_configs("core_without_lag")
    knn = [config for config in configs if config.model_name == "KNN"]
    assert knn
    assert all(config.training_scope == "resource_limited_15000_train_rows" for config in knn)


def test_model_results_have_required_columns() -> None:
    config = benchmark.baseline_configs()[0]
    result = benchmark.empty_result(config, random_state=42, train_rows_available=4, validation_rows=3)
    assert set(benchmark.RUN_COLUMNS).issubset(result)


def test_finalize_selection_does_not_retrain_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("finalize must not train models")

    monkeypatch.setattr(benchmark, "execute_model_config", fail_if_called)
    results = pd.DataFrame([completed_result("winner", "models", mae=1.0)])
    winner = benchmark.select_winning_model(results)
    assert winner["model_run_id"] == "winner"
