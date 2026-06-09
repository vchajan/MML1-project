# Crop Source Reconciliation Summary

- Input rows: 486680
- Legacy rows: 235817
- Expanded rows: 250863
- Canonical rows: 270300
- Model-base rows: 267150

## Source Composition

- Legacy-only keys: 19437
- Expanded-only keys: 34483
- Overlapping keys: 216380
- Corroborated overlaps: 213267
- Conflicting overlaps: 3113
- Unit-corrected conflicts: 29
- Unresolved production-unit conflicts: 787
- Unresolved conflicts with legacy retained: 2297

## Basic Model Exclusions

- Coconut: 2260
- Total foodgrain: 188
- Pulses total: 255
- Oilseeds total: 447
- Total exclusions: 3150

## Validation

- Missing weather values: 0
- Validation checks passed: True
- Canonical parquet: data\interim\crop_weather_canonical_1997_2014.parquet
- Model-base parquet: data\interim\crop_weather_model_base_1997_2014.parquet
