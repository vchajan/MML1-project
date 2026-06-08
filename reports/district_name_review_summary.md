# District Name Anomaly Review

This review uses only local project inputs and exact data comparisons.
No maps were downloaded, no geocoding was performed, no coordinates were assigned, and no fuzzy matching against external data was used.

## Inputs

- Raw crop data SHA-256 before/after: `1A4651D07A271F882869109271610E6E9BD3B1870F3679AE0AC3AAACB728E5BC` / `1A4651D07A271F882869109271610E6E9BD3B1870F3679AE0AC3AAACB728E5BC`
- Required districts SHA-256 before/after: `550D2EF417D736B9DDFA7870A2064A74284EFC440A68F5D01660407237DA8911` / `550D2EF417D736B9DDFA7870A2064A74284EFC440A68F5D01660407237DA8911`
- Crosswalk template SHA-256 before/after: `2A30255FC7AB9B387EDCB6ABAB76EFA8A962C325F2BD2604AB331F25BA8011EF` / `2A30255FC7AB9B387EDCB6ABAB76EFA8A962C325F2BD2604AB331F25BA8011EF`

## Flag Counts

- `flag_name_too_short`: 1
- `flag_single_character_name`: 1
- `flag_contains_digit`: 2
- `flag_contains_parentheses`: 1
- `flag_contains_unusual_punctuation`: 1
- `flag_multiple_spaces_original`: 0
- `flag_leading_or_trailing_space_original`: 0
- `flag_duplicate_normalized_within_state`: 0
- `flag_same_district_name_multiple_states`: 123
- `flag_multiple_raw_names_same_normalized_key`: 0
- `flag_possible_truncation`: 1

## Review Categories

- `parentheses`: 1
- `possible_damage_or_truncation`: 1
- `possible_renaming_or_state_reorganization`: 123
- `punctuation_or_abbreviation`: 3

## Priority Counts

- high: 1
- medium: 127
- low: 0
- normalized conflict rows: 0

## Punjab / S Review

- Result: `strong`; ready_for_crosswalk_override_review; evidence: 73 shared Crop_Year+Season+Crop rows; 55 same Area; 0 same Production; 0 same yield; 0 exact data-row matches; ratio hits x10/x100/x1000 = 17/72/0; candidate name contains one of the requested Punjab review terms
- Top same-state candidate comparisons:

- `HOSHIARPUR`: 132 shared keys, 11 same Area, 12 same Production, 12 same yield, 5 exact matches; status `reviewed_overlap_no_correction`
- `AMRITSAR`: 114 shared keys, 14 same Area, 12 same Production, 11 same yield, 5 exact matches; status `reviewed_overlap_no_correction`
- `GURDASPUR`: 116 shared keys, 6 same Area, 7 same Production, 7 same yield, 4 exact matches; status `reviewed_overlap_no_correction`
- `RUPNAGAR`: 112 shared keys, 7 same Area, 6 same Production, 13 same yield, 4 exact matches; status `name_hint_only_no_data_confirmation`
- `SANGRUR`: 94 shared keys, 9 same Area, 5 same Production, 8 same yield, 4 exact matches; status `reviewed_overlap_no_correction`
- `BARNALA`: 67 shared keys, 13 same Area, 4 same Production, 11 same yield, 4 exact matches; status `reviewed_overlap_no_correction`
- `MOGA`: 66 shared keys, 8 same Area, 7 same Production, 6 same yield, 4 exact matches; status `reviewed_overlap_no_correction`
- `MANSA`: 64 shared keys, 9 same Area, 6 same Production, 5 same yield, 4 exact matches; status `reviewed_overlap_no_correction`

- Requested name-term checks:

- `RUPNAGAR`: 112 shared keys, 4 exact matches; status `name_hint_only_no_data_confirmation`
- `S.A.S NAGAR`: 73 shared keys, 0 exact matches; status `strong_scaled_data_match`

## Manual Overrides

- `Punjab` / `S` -> `S.A.S NAGAR`; type `truncated_name`; confidence `strong`; status `ready_for_crosswalk_override_review`; evidence: 73 shared Crop_Year+Season+Crop rows; 55 same Area; 0 same Production; 0 same yield; 0 exact data-row matches; ratio hits x10/x100/x1000 = 17/72/0; candidate name contains one of the requested Punjab review terms

## Validation

- Raw crop dataset was not modified.
- `required_districts_1997_2014.csv` was not modified.
- All high-priority cases were analyzed.
- Every override row has explicit evidence.
- No district name was automatically corrected using fuzzy similarity.
