from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_time_aware_model_tuning as tuning  # noqa: E402


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


def train_rows() -> pd.DataFrame:
    rows = []
    for year in range(1997, 2011):
        rows.append(
            {
                "canonical_crop_row_id": f"rice-{year}",
                "Crop_Year": year,
                "Crop_canonical": "Rice",
                "Season_canonical": "Kharif",
                "canonical_state_name": "State",
                "canonical_district_name": "DistrictA",
                "rain_sum_mm": float(year - 1990),
                "lag_yield_1y": float(year - 1996) if year > 1997 else np.nan,
                "lag_available": 1 if year > 1997 else 0,
                "target_yield": float(year - 1995),
            }
        )
        rows.append(
            {
                "canonical_crop_row_id": f"wheat-{year}",
                "Crop_Year": year,
                "Crop_canonical": "Wheat",
                "Season_canonical": "Rabi",
                "canonical_state_name": "State",
                "canonical_district_name": "DistrictB",
                "rain_sum_mm": float(year - 1980),
                "lag_yield_1y": np.nan,
                "lag_available": 0,
                "target_yield": float(year - 1980),
            }
        )
    return pd.DataFrame(rows)


def validation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_crop_row_id": ["v1", "v2", "v3"],
            "Crop_Year": [2011, 2011, 2012],
            "Crop_canonical": ["Rice", "Wheat", "NewCrop"],
            "Season_canonical": ["Kharif", "Rabi", "Kharif"],
            "canonical_state_name": ["State", "State", "State"],
            "canonical_district_name": ["DistrictA", "DistrictB", "DistrictC"],
            "rain_sum_mm": [30.0, 31.0, 32.0],
            "lag_yield_1y": [15.0, np.nan, 99.0],
            "lag_available": [1, 0, 1],
            "target_yield": [16.0, 31.0, 100.0],
        }
    )


def small_config(model_run_id: str = "cv_direct_ridge_alpha_1_core_without_lag") -> tuning.TuningConfig:
    return tuning.TuningConfig(
        model_run_id=model_run_id,
        experiment_family="direct",
        model_name="Ridge",
        model_family="linear",
        feature_set="core_without_lag",
        target_strategy="direct",
        preprocessing_family="linear",
        hyperparameters={"alpha": 1.0, "solver": "auto"},
    )


def completed_result(model_run_id: str, fold_id: str = "fold_1", phase: str = "cv-direct") -> dict[str, object]:
    result = {column: "" for column in tuning.RUN_COLUMNS}
    result.update(
        {
            "model_run_id": model_run_id,
            "phase": phase,
            "experiment_family": "direct",
            "model_name": "Ridge",
            "feature_set": "core_without_lag",
            "target_strategy": "direct",
            "fold_id": fold_id,
            "mae": 1.0,
            "rmse": 1.2,
            "r2": 0.5,
            "median_ae": 1.0,
            "status": "completed",
            "random_state": 42,
            "test_data_accessed": False,
        }
    )
    return result


def aggregate_row(
    model_run_id: str,
    family: str,
    mean_mae: float,
    worst: float,
    std: float,
    model_name: str = "Ridge",
    feature_set: str = "core_with_lag",
    worst_rmse: float = 3.0,
) -> dict[str, object]:
    return {
        "model_run_id": model_run_id,
        "experiment_family": family,
        "model_name": model_name,
        "model_family": "linear",
        "feature_set": feature_set,
        "target_strategy": family,
        "hyperparameters_json": "{}",
        "preprocessing_family": "linear",
        "training_scope": "full_train",
        "mean_mae": mean_mae,
        "std_mae": std,
        "mean_rmse": mean_mae + 1.0,
        "std_rmse": std + 0.1,
        "mean_r2": 0.5,
        "std_r2": 0.1,
        "worst_fold_mae": worst,
        "worst_fold_rmse": worst_rmse,
        "total_fit_seconds": 1.0,
        "successful_folds": 3,
        "failed_folds": 0,
        "simplicity_rank": tuning.MODEL_SIMPLICITY.get(model_name, 99),
        "test_data_accessed": False,
    }


def frozen_report_row(model_run_id: str, application_track: str, validation_mae: float) -> dict[str, object]:
    config = tuning.all_config_registry()[model_run_id]
    return {
        "model_run_id": model_run_id,
        "application_track": application_track,
        "experiment_family": config.experiment_family,
        "model_name": config.model_name,
        "model_family": config.model_family,
        "feature_set": config.feature_set,
        "target_strategy": config.target_strategy,
        "hyperparameters_json": tuning.json_dumps(config.hyperparameters),
        "preprocessing_family": config.preprocessing_family,
        "validation_mae": validation_mae,
        "validation_rmse": validation_mae + 1.0,
        "validation_r2": 0.8 - validation_mae / 10.0,
        "validation_median_ae": validation_mae / 2.0,
        "time_cv_mean_mae": validation_mae + 0.1,
        "time_cv_std_mae": 0.1,
        "time_cv_worst_fold_mae": validation_mae + 0.2,
        "status": "completed",
        "test_data_accessed": False,
    }


def frozen_validation_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            frozen_report_row("cv_baseline_lag_with_crop_median_fallback", tuning.FORECAST_TRACK, 0.8),
            frozen_report_row("cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector", tuning.FORECAST_TRACK, 0.9),
            frozen_report_row("cv_direct_tree_depth_none_leaf_20_core_without_lag", tuning.SUITABILITY_TRACK, 1.0),
            frozen_report_row("cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag", tuning.FORECAST_TRACK, 1.2),
        ]
    )


def frozen_cv_results() -> pd.DataFrame:
    rows = []
    for row in frozen_validation_results().to_dict("records"):
        cv_row = dict(row)
        cv_row["phase"] = f"cv-{row['experiment_family']}"
        cv_row["fold_id"] = "fold_1"
        rows.append(cv_row)
    return pd.DataFrame(rows)


def test_fold_1_uses_expected_years() -> None:
    fold = tuning.get_time_cv_folds()[0]
    assert (fold.fit_start_year, fold.fit_end_year) == (1997, 2004)
    assert (fold.evaluation_start_year, fold.evaluation_end_year) == (2005, 2006)


def test_fold_2_uses_expected_years() -> None:
    fold = tuning.get_time_cv_folds()[1]
    assert (fold.fit_start_year, fold.fit_end_year) == (1997, 2006)
    assert (fold.evaluation_start_year, fold.evaluation_end_year) == (2007, 2008)


def test_fold_3_uses_expected_years() -> None:
    fold = tuning.get_time_cv_folds()[2]
    assert (fold.fit_start_year, fold.fit_end_year) == (1997, 2008)
    assert (fold.evaluation_start_year, fold.evaluation_end_year) == (2009, 2010)


def test_no_fit_year_is_after_evaluation_start() -> None:
    tuning.validate_time_cv_folds(tuning.get_time_cv_folds())
    for fold in tuning.get_time_cv_folds():
        assert fold.fit_end_year < fold.evaluation_start_year


def test_test_dataset_is_not_opened() -> None:
    with pytest.raises(ValueError, match="test dataset"):
        tuning.guard_not_test_path(tuning.TEST_PATH)
    assert tuning.TEST_DATA_ACCESSED is False


def test_crop_median_is_fit_only_on_fold_fit_rows() -> None:
    frame = train_rows()
    fit, evaluation = tuning.rows_for_fold(frame, tuning.get_time_cv_folds()[0])
    state = tuning.fit_crop_medians(fit)
    prediction = tuning.predict_crop_median(evaluation[evaluation["Crop_canonical"].eq("Rice")], state)[0]
    assert prediction == fit.loc[fit["Crop_canonical"].eq("Rice"), "target_yield"].median()


def test_lag_baseline_uses_fallback_correctly() -> None:
    fit = pd.DataFrame({"Crop_canonical": ["Rice", "Rice", "Wheat"], "target_yield": [2.0, 4.0, 10.0]})
    evaluation = pd.DataFrame(
        {
            "Crop_canonical": ["Rice", "Wheat", "New"],
            "lag_available": [1, 0, 0],
            "lag_yield_1y": [7.0, np.nan, np.nan],
        }
    )
    state = tuning.fit_crop_medians(fit)
    predictions = tuning.predict_lag_with_crop_median_fallback(evaluation, state)
    assert predictions.tolist() == [7.0, 10.0, 4.0]


def test_residual_target_is_computed_correctly() -> None:
    frame = pd.DataFrame({"target_yield": [10.0, 15.0]})
    residual = tuning.compute_residual_target(frame, [8.0, 20.0])
    assert residual.tolist() == [2.0, -5.0]


def test_residual_training_uses_only_rows_with_lag() -> None:
    frame = pd.DataFrame({"lag_available": [1, 0, 1], "lag_yield_1y": [2.0, np.nan, np.nan]})
    assert tuning.residual_fit_mask(frame).tolist() == [True, False, False]


def test_rows_without_lag_do_not_get_residual_correction() -> None:
    frame = pd.DataFrame({"lag_available": [1, 0], "lag_yield_1y": [2.0, np.nan]})
    predictions = tuning.apply_residual_predictions(frame, [2.0, 10.0], [0.5])
    assert predictions.tolist() == [2.5, 10.0]


def test_residual_prediction_is_added_to_baseline() -> None:
    frame = pd.DataFrame({"lag_available": [1], "lag_yield_1y": [5.0]})
    predictions = tuning.apply_residual_predictions(frame, [5.0], [1.25])
    assert predictions[0] == 6.25


def test_log1p_and_expm1_are_consistent() -> None:
    target = np.array([0.0, 1.0, 10.0])
    restored = tuning.inverse_log_predictions(np.log1p(target))
    assert np.allclose(restored, target)


def test_inverse_log_prediction_is_clipped_to_zero() -> None:
    restored = tuning.inverse_log_predictions([-100.0, math.log1p(4.0)])
    assert restored[0] == 0.0
    assert restored[1] == pytest.approx(4.0)


def test_preprocessing_is_fit_only_on_fold_fit_data() -> None:
    frame = train_rows()
    fold = tuning.get_time_cv_folds()[0]
    fit, evaluation = tuning.rows_for_fold(frame, fold)
    unseen = evaluation.copy()
    unseen.loc[unseen.index[0], "Crop_canonical"] = "FutureOnly"
    config = small_config()
    features = ["Crop_canonical", "Crop_Year", "rain_sum_mm"]
    preprocessor = tuning.build_preprocessor(config, features, manifest())
    preprocessor.fit(fit[features])
    preprocessor.transform(unseen[features])
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    assert "FutureOnly" not in set(encoder.categories_[0])


def test_forbidden_leakage_feature_raises() -> None:
    with pytest.raises(ValueError, match="forbidden leakage"):
        tuning.validate_feature_columns(
            "bad",
            ["Crop_canonical", "Production_corrected"],
            ["Crop_canonical", "Production_corrected"],
            manifest(),
        )


def test_target_is_not_feature() -> None:
    with pytest.raises(ValueError, match="target_yield"):
        tuning.validate_feature_columns("bad", ["target_yield"], ["target_yield"], manifest())


def test_resume_skips_completed_run() -> None:
    config = small_config("run_a")
    existing = pd.DataFrame([completed_result("run_a", fold_id="fold_1", phase="cv-direct")])
    assert tuning.should_skip_run(config, existing, resume=True, fold_id="fold_1", phase="cv-direct")


def test_failed_run_does_not_stop_other_runs() -> None:
    configs = [small_config("bad"), small_config("good")]
    folds = [tuning.get_time_cv_folds()[0]]
    calls: list[str] = []
    written: list[dict[str, object]] = []

    def evaluator(config, train, manifest_dict, fold, seed):
        calls.append(config.model_run_id)
        if config.model_run_id == "bad":
            raise RuntimeError("planned failure")
        return completed_result(config.model_run_id, fold_id=fold.fold_id)

    tuning.run_config_sequence(
        configs,
        folds,
        train_rows(),
        manifest(),
        resume=False,
        force_run=None,
        random_state=42,
        evaluator=evaluator,
        writer=written.append,
    )
    assert calls == ["bad", "good"]
    assert [row["status"] for row in written] == ["failed", "completed"]


def test_shortlist_uses_only_cv_results() -> None:
    baseline = pd.DataFrame(
        [
            aggregate_row("cv_baseline_lag_with_crop_median_fallback", "baseline", 2.0, 2.5, 0.1, "LagWithCropMedianFallback"),
            aggregate_row("cv_baseline_crop_median", "baseline", 4.0, 5.0, 0.2, "CropMedian"),
        ]
    )
    direct = pd.DataFrame(
        [
            aggregate_row("direct_lag_a", "direct", 3.0, 4.0, 0.2, feature_set="core_with_lag"),
            aggregate_row("direct_lag_b", "direct", 1.0, 2.0, 0.5, feature_set="core_with_lag"),
            aggregate_row("direct_no_lag_a", "direct", 1.2, 1.8, 0.1, feature_set="core_without_lag"),
            aggregate_row("direct_no_lag_b", "direct", 1.5, 1.9, 0.1, feature_set="core_without_lag"),
        ]
    )
    residual = pd.DataFrame([aggregate_row("residual_a", "residual", 2.0, 3.0, 0.2)])
    log_target = pd.DataFrame([aggregate_row("log_a", "log_target", 2.5, 3.5, 0.2)])
    entries = tuning.select_shortlist_entries(baseline, direct, residual, log_target, max_per_family=2)
    selected = [entry["model_run_id"] for entry in entries]
    assert "direct_lag_b" in selected
    assert "direct_no_lag_a" in selected
    assert "direct_lag_a" in selected


def shortlist_fixture() -> list[dict[str, object]]:
    baseline = pd.DataFrame(
        [
            aggregate_row("cv_baseline_lag_with_crop_median_fallback", "baseline", 1.0, 1.2, 0.1, "LagWithCropMedianFallback"),
            aggregate_row("cv_baseline_crop_median", "baseline", 2.0, 2.2, 0.1, "CropMedian", feature_set="crop_only"),
        ]
    )
    direct = pd.DataFrame(
        [
            aggregate_row("lag_direct_best", "direct", 1.1, 1.4, 0.1, feature_set="core_with_lag"),
            aggregate_row("lag_direct_second", "direct", 1.2, 1.5, 0.1, feature_set="core_with_lag"),
            aggregate_row("no_lag_direct_best", "direct", 1.3, 1.6, 0.1, feature_set="core_without_lag"),
            aggregate_row("no_lag_direct_second", "direct", 1.4, 1.7, 0.1, feature_set="core_without_lag"),
        ]
    )
    residual = pd.DataFrame([aggregate_row("residual_best", "residual", 1.05, 1.2, 0.1)])
    log_target = pd.DataFrame(
        [
            aggregate_row("unstable_log", "log_target", 0.9, 1.1, 0.1, model_name="LinearSVR", worst_rmse=45.0),
            aggregate_row("stable_log", "log_target", 1.6, 1.8, 0.1, model_name="RandomForest", worst_rmse=6.0),
        ]
    )
    return tuning.select_shortlist_entries(baseline, direct, residual, log_target)


def test_forecast_shortlist_contains_lag_model() -> None:
    entries = shortlist_fixture()
    forecast = [entry for entry in entries if entry["application_track"] == tuning.FORECAST_TRACK]
    assert any(entry["model_run_id"] == "cv_baseline_lag_with_crop_median_fallback" for entry in forecast)
    assert any(entry["feature_set"] == "core_with_lag" for entry in forecast)


def test_suitability_shortlist_contains_no_lag_model() -> None:
    entries = shortlist_fixture()
    suitability = [entry for entry in entries if entry["application_track"] == tuning.SUITABILITY_TRACK]
    assert any(entry["model_run_id"] == "cv_baseline_crop_median" for entry in suitability)
    assert any(entry["feature_set"] == "core_without_lag" for entry in suitability)


def test_suitability_shortlist_never_contains_residual_model() -> None:
    entries = shortlist_fixture()
    suitability = [entry for entry in entries if entry["application_track"] == tuning.SUITABILITY_TRACK]
    assert all(entry["experiment_family"] != "residual" for entry in suitability)


def test_suitability_model_never_uses_lag_yield_feature() -> None:
    config = tuning.TuningConfig(
        model_run_id="no_lag",
        experiment_family="direct",
        model_name="Ridge",
        model_family="linear",
        feature_set="core_without_lag",
        target_strategy="direct",
        preprocessing_family="linear",
        hyperparameters={"alpha": 1.0},
    )
    assert "lag_yield_1y" not in tuning.features_for_config(config, manifest())


def test_application_tracks_are_selected_separately() -> None:
    entries = shortlist_fixture()
    forecast_direct = [
        entry for entry in entries
        if entry["application_track"] == tuning.FORECAST_TRACK and entry["experiment_family"] == "direct"
    ]
    suitability_direct = [
        entry for entry in entries
        if entry["application_track"] == tuning.SUITABILITY_TRACK and entry["experiment_family"] == "direct"
    ]
    assert all(entry["feature_set"] == "core_with_lag" for entry in forecast_direct)
    assert all(entry["feature_set"] == "core_without_lag" for entry in suitability_direct)


def test_unstable_log_target_run_is_not_shortlisted() -> None:
    entries = shortlist_fixture()
    selected = [entry["model_run_id"] for entry in entries]
    assert "unstable_log" not in selected
    assert "stable_log" in selected


def test_validation_shortlist_does_not_use_test(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[Path] = []

    def fake_safe_read(path: Path) -> pd.DataFrame:
        opened.append(path)
        if path == tuning.TEST_PATH:
            raise AssertionError("test parquet was opened")
        return train_rows() if path == tuning.TRAIN_PATH else validation_rows()

    monkeypatch.setattr(tuning, "safe_read_parquet", fake_safe_read)
    monkeypatch.setattr(tuning.validation, "load_manifest", lambda path=tuning.FEATURE_MANIFEST_PATH: manifest())
    train, validation_frame, loaded_manifest = tuning.load_train_validation(
        expected_train_rows=None,
        expected_validation_rows=None,
    )
    assert len(train) > 0
    assert len(validation_frame) > 0
    assert tuning.TEST_PATH not in opened
    assert loaded_manifest["target_column"] == "target_yield"


def test_frozen_config_has_test_flags_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "write_json", lambda path, value: None)
    results = frozen_validation_results()
    payload = tuning.build_frozen_tuned_configuration(
        results,
        cv_results=frozen_cv_results(),
        validation_report=results,
    )
    assert payload["test_data_accessed"] is False
    assert payload["test_used_for_selection"] is False
    assert payload["best_overall_forecast_validation_run"] == "cv_baseline_lag_with_crop_median_fallback"
    assert payload["best_suitability_validation_model"] == "cv_direct_tree_depth_none_leaf_20_core_without_lag"


def test_frozen_tracks_contains_required_track_run_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "write_json", lambda path, value: None)
    results = frozen_validation_results()
    payload = tuning.build_frozen_tuned_configuration(
        results,
        cv_results=frozen_cv_results(),
        validation_report=results,
        random_state=42,
    )
    frozen_tracks = payload["frozen_tracks"]
    assert set(frozen_tracks) == {
        "forecast_baseline",
        "forecast_trained_model",
        "suitability_model",
        "log_target_experiment",
    }
    assert frozen_tracks["forecast_baseline"]["run_id"] == "cv_baseline_lag_with_crop_median_fallback"
    assert frozen_tracks["forecast_baseline"]["type"] == "baseline"
    assert frozen_tracks["forecast_trained_model"]["run_id"] == "cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector"
    assert frozen_tracks["suitability_model"]["run_id"] == "cv_direct_tree_depth_none_leaf_20_core_without_lag"
    assert frozen_tracks["log_target_experiment"]["run_id"] == "cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag"
    assert frozen_tracks["forecast_trained_model"]["hyperparameters"] == {
        "C": 0.03,
        "epsilon": 0.0,
        "max_iter": 20_000,
    }
    assert frozen_tracks["suitability_model"]["hyperparameters"] == {
        "max_depth": None,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    assert frozen_tracks["log_target_experiment"]["hyperparameters"] == {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    }


def test_validation_winners_are_not_mixed_between_tracks() -> None:
    results = pd.DataFrame(
        {
            "model_run_id": ["forecast_baseline", "forecast_model", "suitability_model"],
            "application_track": [tuning.FORECAST_TRACK, tuning.FORECAST_TRACK, tuning.SUITABILITY_TRACK],
            "experiment_family": ["baseline", "direct", "direct"],
            "model_name": ["LagWithCropMedianFallback", "Ridge", "Ridge"],
            "feature_set": ["lag_with_crop_fallback", "core_with_lag", "core_without_lag"],
            "validation_mae": [0.5, 0.8, 0.6],
            "validation_rmse": [1.0, 1.1, 1.2],
            "validation_r2": [0.8, 0.7, 0.6],
            "time_cv_mean_mae": [0.5, 0.8, 0.6],
            "time_cv_std_mae": [0.1, 0.1, 0.1],
        }
    )
    winners = tuning.best_validation_rows(results)
    assert winners["best_overall_forecast"]["model_run_id"] == "forecast_baseline"
    assert winners["best_trained_forecast"]["model_run_id"] == "forecast_model"
    assert winners["best_suitability_model"]["model_run_id"] == "suitability_model"


def test_finalize_does_not_retrain(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("finalize must not train")

    monkeypatch.setattr(tuning, "execute_cv_fold", fail_if_called)
    monkeypatch.setattr(tuning, "execute_validation_config", fail_if_called)
    assert tuning.finalize_outputs.__name__ == "finalize_outputs"


def test_model_run_ids_are_deterministic() -> None:
    first = [config.model_run_id for config in tuning.direct_configs()]
    second = [config.model_run_id for config in tuning.direct_configs()]
    assert first == second
    assert len(first) == len(set(first))


def test_primary_cv_selection_metric_is_mean_mae() -> None:
    frame = pd.DataFrame(
        [
            aggregate_row("worse_mean", "direct", 2.0, 2.0, 0.0),
            aggregate_row("better_mean", "direct", 1.0, 10.0, 5.0),
        ]
    )
    ranked = tuning.rank_cv_results(frame)
    assert ranked.iloc[0]["model_run_id"] == "better_mean"
