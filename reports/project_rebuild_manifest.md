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
- Legacy source has priority for 3,113 conflicting overlapping keys.
- Coconut and aggregate crop categories are documented and excluded from the basic model dataset, but retained in the complete canonical dataset.
- Weather features were preserved from the existing crop-weather dataset; weather aggregation was not recomputed during source reconciliation.
- Source reconciliation rules are stored in `data/reference/crop_source_reconciliation_rules.json`.
- Chronological modeling dataset and train/validation/test splits completed from the local model-base Parquet.
- Feature sets are recorded in `data/reference/model_feature_manifest.json`.
- One-year lag yield is built with an explicit self-join on the same district, crop and season with `Crop_Year = Y - 1`.
- No preprocessing, target outlier treatment, baseline model, hyperparameter tuning or test evaluation has been performed.

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
- Coconut exclusions: 2,260
- Aggregate-category exclusions: 890
- Missing weather values after reconciliation: 0

## Modeling Dataset Counts

- Input model-base rows: 267,150
- Model dataset rows: 267,150
- Train rows, 1997-2010: 202,166
- Validation rows, 2011-2012: 32,388
- Test rows, 2013-2014: 32,596
- Rows with `lag_yield_1y`: 213,187
- Rows without `lag_yield_1y`: 53,963
- Train rows with lag: 156,369
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

## Next Step

Fit preprocessing and any target outlier treatment only on the train period, then build baseline models. The 2013-2014 test split must stay unused until final evaluation.
