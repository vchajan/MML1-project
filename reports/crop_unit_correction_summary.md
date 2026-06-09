# Crop Unit Correction Summary

- Conflict pairs reviewed: 3113
- Area-unit corrections applied: 29
- Unresolved production-unit conflicts: 787
- Unresolved conflicts with legacy retained: 2297

Corrections are based on source-pair evidence only. No absolute target threshold, clipping, winsorization or row deletion is used.

## Punjab / 2011 / Whole Year / Sugarcane

- Conflict rows: 15
- Legacy Area range: 1..20
- Expanded Area range: 1000..20000
- Area ratio patterns: 1000
- Corrected production ratio patterns: 1
- Legacy target range: 40000..88000
- Expanded target range: 40..88
- Corrected productions match: True
- Selected source after rule: expanded_source_x100
- Status after rule: conflict_unit_corrected

### Punjab Focus Districts

- GURDASPUR: legacy_target=74550; expanded_target=74.55; corrected_target=74.55; selected_source=expanded_source_x100
- PATIALA: legacy_target=88000; expanded_target=88; corrected_target=88; selected_source=expanded_source_x100
- S.A.S NAGAR: legacy_target=65000; expanded_target=65; corrected_target=65; selected_source=expanded_source_x100
- TARN TARAN: legacy_target=40000; expanded_target=40; corrected_target=40; selected_source=expanded_source_x100

## Tamil Nadu / 1997 / Whole Year / Sugarcane

- Conflict rows: 0
- Conclusion: not present in the 3,113 conflicting source pairs.

## Validation Checks

- canonical_row_count_preserved: True (observed=270300; expected=270300)
- model_base_row_count_preserved: True (observed=267150; expected=267150)
- conflict_report_rows: True (observed=3113; expected=3113)
- unit_corrected_conflicts: True (observed=29; expected=29)
- punjab_2011_sugarcane_corrected_rows: True (rows=15; statuses=['conflict_unit_corrected'])
- punjab_2011_sugarcane_target_scale: True (corrected_target_range=40..88)
- tamil_nadu_1997_sugarcane_conflict_pairs: True (conflict_rows=0; statuses=[])
