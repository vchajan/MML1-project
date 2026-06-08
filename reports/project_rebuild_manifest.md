# Project Rebuild Manifest

Date: 2026-06-08

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

The large interim CSV remains local and ignored by Git.

## Rebuild Steps

- District-name audit completed; geographic matching not yet performed.
- District-name anomaly review completed; no geographic matching or coordinate assignment has been performed yet.

## Next Step

Create the district crosswalk and verified district points.
