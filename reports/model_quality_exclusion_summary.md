# Model Quality Exclusion Summary

Two source-corroborated records are retained in the full canonical and model-base interim datasets, but excluded from the modeling dataset before lag feature construction.

- Canonical/model-base rows unchanged by this script: true
- Modeling rows before exclusion: 267150
- Modeling rows after exclusion: 267148
- Excluded modeling rows: 2
- Lag rows affected by removing the source rows before self-join: 1

## Excluded Rows

| canonical_crop_row_id | canonical_state_name | canonical_district_name | Crop_Year | Season_canonical | Crop_canonical | Area | Area_corrected | Production | Production_corrected | target_yield | exclusion_scope | exclusion_reason | evidence | identified_from_period | review_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCR_D813C3DC43AF694A5EF8 | Haryana | KARNAL | 2008 | Whole Year | Onion | 2.0 | 2.0 | 4300.0 | 4300.0 | 2150.0 | modeling_only | Source-corroborated value is incompatible with the same district/crop time series; exact corrected value cannot be proven. | Haryana / KARNAL / 2008 / Whole Year / Onion has Area=2 Production=4300 target_yield=2150 while neighboring same district/crop years are around 14-27. | train_1997_2010 | confirmed_source_record_corruption |
| CCR_1D8AB7669408410FDFD9 | Tamil Nadu | PERAMBALUR | 2008 | Whole Year | Cashewnut | 1.0 | 1.0 | 9801.0 | 9801.0 | 9801.0 | modeling_only | Source-corroborated Area/Production combination is incompatible with adjacent years and propagates an invalid one-year lag; exact corrected value cannot be proven. | Tamil Nadu / PERAMBALUR / 2008 / Whole Year / Cashewnut has Area=1 Production=9801 target_yield=9801 and creates invalid 2009 lag_yield_1y=9801. | train_1997_2010 | confirmed_source_record_corruption |

## Lag Availability/Source Changes

| canonical_crop_row_id | Crop_Year | canonical_state_name | canonical_district_name | Crop_canonical | Season_canonical | lag_available_before | lag_available_after | lag_yield_1y_before | lag_yield_1y_after | lag_source_crop_year_before | lag_source_crop_year_after | lag_source_canonical_crop_row_id_before | lag_source_canonical_crop_row_id_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CCR_D28BD19892E28C79012D | 2009 | Tamil Nadu | PERAMBALUR | Cashewnut | Whole Year | 1 | 0 | 9801.0 |  | 2008 |  | CCR_1D8AB7669408410FDFD9 |  |

No numeric target correction, winsorization, target-threshold filtering, validation target analysis, or test target analysis was performed.
