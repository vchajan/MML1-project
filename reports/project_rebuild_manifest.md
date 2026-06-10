# Project Rebuild Manifest

Date: 2026-06-09

Original HW2 commit SHA: `318dd27a043afe7b661ec5535d42d475f7d33644`

Archive tag: `hw2-original`

Working branch: `rebuild/crop-weather`

## Files Moved To External Archive

External archive: `C:\Users\ASUS\Desktop\MML\MML1-local-archive`

- `HW2.pdf`
- `all_district_coordinates.csv`
- `create_hw2_repo.py`
- `create_hw2_repo_final_clean.py`
- `create_hw2_repo_fixed.py`
- `crop_regions_cleaned.csv`
- `crop_with_coords_final.csv`
- `crop_yield.csv`
- `crop_yield.csv.zip`
- `get_coords.py`
- `stiahni_pocasie.py`
- `weather_data_final.csv`
- `weather_features_final_pipeline_with_coords.zip`
- `Indian_crop_production_yield_dataset_old_4493186c.csv`

The archived `Indian_crop_production_yield_dataset_old_4493186c.csv` is the old modified `data/raw/Indian_crop_production_yield_dataset.csv` with SHA-256 `4493186C89A1B9049E07FD5E2E364ED6D99939DDE8E6ECB56A9553C36B844C87`.

## Authoritative Crop Dataset

- `data/raw/Indian_crop_production_yield_dataset.csv`
- SHA-256: `1A4651D07A271F882869109271610E6E9BD3B1870F3679AE0AC3AAACB728E5BC`

## New Files Placed In Project

- `data/reference/crop_calendar_rules_1997_2014_v1.csv`
- `data/reference/required_districts_1997_2014.csv`
- `data/reference/district_crosswalk_template_1997_2014.csv`
- `data/interim/crop_with_calendar_dates_1997_2014.csv`
- `reports/crop_calendar_application_validation.csv`
- `reports/crop_calendar_application_summary.txt`
- `reports/district_requirements_summary.txt`

## Old Tracked Files Removed From Rebuild Branch

- `benchmark.html`
- `benchmark.ipynb`
- `data/benchmark_validation_results.csv`
- `data/crop_weather_joined.csv`
- `data/final_dataset_weather_complete.csv`
- `data/raw/crop_regions_cleaned.csv`
- `data/raw/crop_with_coords_final.csv`
- `data/test.csv`
- `data/train.csv`
- `data/validation.csv`
- `data/weather_data_final.csv`
- `dataprocessing.html`
- `dataprocessing.ipynb`
- `reports/final_dataset_summary.txt`
- `reports/notes_to_paste.md`
- `src/build_weather_data.py`
- `src/finalize_hw2.py`
- `src/join_crop_weather.py`

## Missing Expected Files

- None.

## Large Local Ignored Files

- `data/interim/crop_with_calendar_dates_1997_2014.csv`
- `data/interim/weather_daily/`
- `data/interim/weather_features_by_window_1997_2014.parquet`
- `data/interim/crop_weather_dataset_1997_2014.parquet`
- `data/interim/crop_weather_canonical_1997_2014.parquet`
- `data/interim/crop_weather_model_base_1997_2014.parquet`
- `data/processed/model_dataset_1997_2014.parquet`
- `data/processed/train_1997_2010.parquet`
- `data/processed/validation_2011_2012.parquet`
- `data/processed/test_2013_2014.parquet`

The large interim CSV, NASA POWER cache, and derived interim/processed Parquet files remain local and ignored by Git.

## Rebuild Steps

- District-name audit completed.
- District-name anomaly review completed.
- District name override stage completed.
- DataMeet Census 2001 and Census 2011 district boundary layers downloaded and technically audited at source commit `b3fbbde595310b397a55d718e0958ce249a4fa1f`; raw map files remain local and ignored by Git, and source/license notes are in `data/reference/boundary_sources/datameet_district_boundaries.json`.
- Name matching against Census 2001 and Census 2011 boundary inventories completed; fuzzy matching remains marked in assignment confidence fields.
- Working historical district points completed for 727 districts and crop years 1997-2014.
- NASA POWER daily weather download completed for 701 weather points, with 701 cache CSV files and 0 failed points.
- Weather aggregation completed for 150,832 weather windows; all windows are valid and minimum coverage is 1.000000.
- Crop-weather join completed with 486,680 input rows and 486,680 output rows.
- Crop source reconciliation completed. The raw crop dataset contains `legacy_source` and `expanded_source_x100`; expanded rows are normalized by factor `0.01`.
- Canonical source key reconciliation completed with 270,300 canonical rows.
- Basic model dataset prepared with 267,150 rows.
- Conflict source-pair unit diagnostics completed for 3,113 conflicting overlapping keys.
- Deterministic source-pair area-unit corrections applied to 29 conflicts.
- Unresolved production-unit conflicts remain marked for 787 conflicts.
- Other unresolved conflicts retain legacy source for 2,297 conflicts.
- Coconut and aggregate crop categories are documented and excluded from the basic model dataset, but retained in the complete canonical dataset.
- Weather features were preserved from the existing crop-weather dataset; weather aggregation was not recomputed during source reconciliation.
- Source reconciliation rules are stored in `data/reference/crop_source_reconciliation_rules.json`.
- Chronological modeling dataset and train/validation/test splits completed from the local model-base Parquet.
- Two confirmed source-corroborated but internally corrupted train-period records are retained in the full canonical/model-base datasets and excluded only from the processed modeling dataset before lag self-join.
- Feature sets are recorded in `data/reference/model_feature_manifest.json`.
- One-year lag yield is built with an explicit self-join on the same district, crop and season with `Crop_Year = Y - 1`.
- Validation benchmark completed using only train 1997-2010 and validation 2011-2012.
- Baseline models were compared on validation data.
- Feature set selection used validation metrics only and selected `core_without_lag`.
- The winning validation configuration was frozen in `data/reference/frozen_model_configuration.json`.
- Time-aware model tuning completed using only train years 1997-2010 for expanding-window CV.
- Time-CV shortlist was evaluated once on validation years 2011-2012.
- Forecast and suitability application tracks were selected separately.
- The winning trained forecast configuration was frozen in `data/reference/frozen_tuned_model_configuration.json`.
- No target outlier treatment or final test evaluation has been performed.
- The 2013-2014 test split was not loaded or used for preprocessing, feature selection, hyperparameter tuning, model selection or evaluation.
- The processed modeling dataset has been regenerated after unit corrections and modeling-only quality exclusions, and the time-aware tuning artifacts now reflect the regenerated modeling dataset.

## Crop Source Reconciliation Counts

- Input rows: 486,680
- Legacy rows: 235,817
- Expanded rows: 250,863
- Canonical rows: 270,300
- Model-base rows: 267,150
- Legacy-only keys: 19,437
- Expanded-only keys: 34,483
- Overlapping keys: 216,380
- Corroborated overlaps: 213,267
- Conflicting overlaps: 3,113
- Unit-corrected conflicts: 29
- Unresolved production-unit conflicts: 787
- Unresolved conflicts with legacy retained: 2,297
- Coconut exclusions: 2,260
- Aggregate-category exclusions: 890
- Missing weather values after reconciliation: 0

## Crop Unit Correction Results

- Conflict pairs reviewed: 3,113
- Pattern groups: 329
- Area-unit corrections applied: 29
- Punjab 2011 / Whole Year / Sugarcane corrections: 15
- Punjab 2011 corrected target range: 40..88
- Punjab focus values:
  - Gurdaspur: 74,550 -> 74.55
  - Patiala: 88,000 -> 88
  - S.A.S NAGAR: 65,000 -> 65
  - Tarn Taran: 40,000 -> 40
- Tamil Nadu 1997 / Whole Year / Sugarcane conflict rows: 0
- Correction rule: source-pair evidence only; no absolute target threshold, clipping, winsorization or row deletion.
- Canonical validation checks passed: true

Unit correction reports:

- `reports/crop_unit_conflict_patterns.csv`
- `reports/crop_unit_conflict_details.csv`
- `reports/crop_unit_conflict_summary.md`
- `reports/crop_unit_corrections_applied.csv`
- `reports/crop_unit_correction_validation.csv`
- `reports/crop_unit_correction_summary.md`

## Modeling Dataset Counts

- Input model-base rows: 267,150
- Model dataset rows before quality exclusions: 267,150
- Modeling-only quality exclusions: 2
- Model dataset rows: 267,148
- Train rows, 1997-2010: 202,164
- Validation rows, 2011-2012: 32,388
- Test rows, 2013-2014: 32,596
- Rows with `lag_yield_1y`: 213,184
- Rows without `lag_yield_1y`: 53,964
- Train rows with lag: 156,366
- Validation rows with lag: 28,001
- Test rows with lag: 28,817
- Feature columns without lag: 31
- Feature columns with lag: 33
- Categorical features: 4
- Numeric core features: 4
- Weather features: 23
- Missing target values: 0
- Missing weather feature values: 0
- Validation unseen categories: 19
- Test unseen categories: 43

Generated reports:

- `reports/modeling_dataset_summary.md`
- `reports/chronological_split_validation.csv`
- `reports/modeling_feature_schema.csv`
- `reports/modeling_unseen_categories.csv`
- `reports/modeling_lag_summary.csv`
- `reports/modeling_dataset_sample.csv`
- `reports/model_quality_exclusions.csv`
- `reports/model_quality_exclusion_summary.md`

## Model Quality Exclusions

- Exclusion scope: `modeling_only`
- Reference file: `data/reference/model_quality_exclusions.csv`
- Excluded IDs:
  - `CCR_D813C3DC43AF694A5EF8` - Haryana / KARNAL / 2008 / Whole Year / Onion
  - `CCR_1D8AB7669408410FDFD9` - Tamil Nadu / PERAMBALUR / 2008 / Whole Year / Cashewnut
- Both rows remain in the full canonical and model-base interim datasets.
- Exclusions are applied before lag self-join in the processed modeling dataset.
- No numeric target correction, winsorization, general target threshold or validation/test target analysis was used.
- Lag rows affected: 1 (`CCR_D28BD19892E28C79012D`, Tamil Nadu / PERAMBALUR / 2009 / Whole Year / Cashewnut lost invalid `lag_yield_1y = 9801`).

## Time-Aware Model Tuning Results

Time-aware model selection uses expanding-window CV only inside train years 1997-2010:

- Fold 1: fit 1997-2004, evaluate 2005-2006
- Fold 2: fit 1997-2006, evaluate 2007-2008
- Fold 3: fit 1997-2008, evaluate 2009-2010
- Rolling lag assumption: previous-year official yield is available when predicting the following year.
- Test data accessed: false
- Test used for selection: false

CV winners:

- Best CV baseline: `cv_baseline_lag_with_crop_median_fallback`, mean MAE 1.074877
- Best direct CV model: `cv_direct_rf_200_depth_none_leaf_20_maxfeat_0_5_core_with_lag`, mean MAE 1.259570
- Best residual CV model: `cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector`, mean MAE 1.083049
- Best log-target CV model: `cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag`, mean MAE 1.578012

Validation shortlist winners:

- Best overall forecast validation run: `cv_baseline_lag_with_crop_median_fallback`, validation MAE 1.233652
- Best trained forecast validation model: `cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector`, validation MAE 1.240889
- Best suitability validation model: `cv_direct_tree_depth_none_leaf_20_core_without_lag`, validation MAE 1.673477

Application tracks:

- `forecast_with_lag` includes lag baseline, direct `core_with_lag`, residual lag-corrector and stable log-target candidates.
- `suitability_without_lag` includes crop median baseline and direct `core_without_lag` candidates only.
- Residual models did not beat the lag baseline overall on validation MAE, but the best residual LinearSVR is the best trained forecast model.
- Log-target models did not improve over the best direct/residual candidates.

Time-aware artifacts:

- `src/run_time_aware_model_tuning.py`
- `tests/test_time_aware_model_tuning.py`
- `data/reference/time_cv_shortlist.json`
- `data/reference/frozen_tuned_model_configuration.json`
- `reports/time_cv_baseline_fold_results.csv`
- `reports/time_cv_direct_results.csv`
- `reports/time_cv_residual_results.csv`
- `reports/time_cv_log_target_results.csv`
- `reports/time_cv_all_results.csv`
- `reports/tuned_validation_results.csv`
- `reports/tuned_validation_runtime.csv`
- `reports/tuned_validation_predictions_sample.csv`
- `reports/tuned_validation_comparison.csv`
- `reports/tuned_validation_subgroup_metrics.csv`
- `reports/time_aware_tuning_summary.md`
- `reports/time_cv_mae.png`
- `reports/time_cv_stability.png`
- `reports/tuned_validation_mae.png`

## Next Step

Run final test evaluation as a separate step only after accepting the frozen time-aware configuration. The 2013-2014 test split remains unused for model selection.
