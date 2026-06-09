# Validation Benchmark Summary

- Train rows: 202166
- Validation rows: 32388
- Test data accessed: false
- Final test evaluation: not performed
- Selected feature set: `core_without_lag`
- Best model: `tree_depth_none_leaf_20_core_without_lag`
- Best baseline: `baseline_crop_median`
- Absolute MAE improvement over baseline: 0.679662
- Relative MAE improvement over baseline: 1.93%

Overfitting cannot be determined definitively from validation results alone.
KNN uses resource-limited training with at most 15,000 train rows, so it is not fully directly comparable to full-train models.

## Baselines

| model_run_id | mae | rmse | r2 | median_ae |
| --- | --- | --- | --- | --- |
| baseline_global_median | 37.025923758194246 | 1548.118510562045 | -0.0005605902856424 | 0.6332223165992091 |
| baseline_crop_median | 35.271541421263834 | 1546.982880298817 | 0.0009068040017777 | 0.4335613552671066 |
| baseline_lag_with_crop_median_fallback | 67.08796216279376 | 2187.430084473654 | -0.997576166867146 | 0.1815815060592565 |

## Feature Set Comparison

| model_run_id | feature_set | model_name | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| feature_set_ridge_core_without_lag | core_without_lag | Ridge | 36.13711115536729 | 1546.855553226335 | 0.0010712613829406 |
| feature_set_decision_tree_core_without_lag | core_without_lag | DecisionTree | 34.69140748959207 | 1546.735964365898 | 0.001225711656527 |
| feature_set_ridge_core_with_lag | core_with_lag | Ridge | 37.315304940347545 | 1548.0028141531393 | -0.0004110449623704 |
| feature_set_decision_tree_core_with_lag | core_with_lag | DecisionTree | 34.587314125810515 | 1546.764922255973 | 0.0011883133319782 |

## All Successful Models

| model_run_id | phase | model_name | feature_set | mae | rmse | r2 | training_scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_crop_median | baselines | CropMedian | crop_only | 35.271541421263834 | 1546.982880298817 | 0.0009068040017777 | full_train |
| baseline_global_median | baselines | GlobalMedian | none | 37.025923758194246 | 1548.118510562045 | -0.0005605902856424 | full_train |
| baseline_lag_with_crop_median_fallback | baselines | LagWithCropMedianFallback | lag_with_crop_fallback | 67.08796216279376 | 2187.430084473654 | -0.997576166867146 | full_train |
| feature_set_decision_tree_core_with_lag | feature-sets | DecisionTree | core_with_lag | 34.587314125810515 | 1546.764922255973 | 0.0011883133319782 | full_train |
| feature_set_decision_tree_core_without_lag | feature-sets | DecisionTree | core_without_lag | 34.69140748959207 | 1546.735964365898 | 0.001225711656527 | full_train |
| feature_set_ridge_core_without_lag | feature-sets | Ridge | core_without_lag | 36.13711115536729 | 1546.855553226335 | 0.0010712613829406 | full_train |
| feature_set_ridge_core_with_lag | feature-sets | Ridge | core_with_lag | 37.315304940347545 | 1548.0028141531393 | -0.0004110449623704 | full_train |
| tree_depth_none_leaf_20_core_without_lag | models | DecisionTree | core_without_lag | 34.59187898314 | 1546.7838409609774 | 0.0011638799627642 | full_train |
| tree_depth_15_leaf_10_core_without_lag | models | DecisionTree | core_without_lag | 34.69140748959207 | 1546.735964365898 | 0.001225711656527 | full_train |
| tree_depth_8_leaf_20_core_without_lag | models | DecisionTree | core_without_lag | 35.088383555362206 | 1546.9302811555142 | 0.0009747434126847 | full_train |
| linearsvr_c_10_core_without_lag | models | LinearSVR | core_without_lag | 35.189818929036925 | 1546.967340660423 | 0.0009268759356146 | full_train |
| linearsvr_c_1_core_without_lag | models | LinearSVR | core_without_lag | 35.19541941631007 | 1546.971756921109 | 0.0009211716475628 | full_train |
| linearsvr_c_0_1_core_without_lag | models | LinearSVR | core_without_lag | 35.21298007265341 | 1547.0115487024943 | 0.0008697736340642 | full_train |
| rf_200_depth_none_leaf_10_core_without_lag | models | RandomForest | core_without_lag | 35.213870232560176 | 1547.2255427170091 | 0.000593340434736 | full_train |
| rf_150_depth_20_leaf_5_core_without_lag | models | RandomForest | core_without_lag | 35.218510669926985 | 1547.1691807025163 | 0.0006661514698594 | full_train |
| lasso_alpha_0_001_core_without_lag | models | Lasso | core_without_lag | 36.05391750214343 | 1546.8543168118108 | 0.0010728582864193 | full_train |
| ridge_alpha_10_core_without_lag | models | Ridge | core_without_lag | 36.11389040860078 | 1546.857261899563 | 0.0010690545269331 | full_train |
| lasso_alpha_0_0001_core_without_lag | models | Lasso | core_without_lag | 36.12530238633336 | 1546.854530519368 | 0.0010725822703893 | full_train |
| ridge_alpha_1_core_without_lag | models | Ridge | core_without_lag | 36.13711115536729 | 1546.855553226335 | 0.0010712613829406 | full_train |
| ridge_alpha_0_1_core_without_lag | models | Ridge | core_without_lag | 36.14074096005942 | 1546.8554859507378 | 0.0010713482734396 | full_train |
| linear_regression_core_without_lag | models | LinearRegression | core_without_lag | 36.14333895039693 | 1546.855344983872 | 0.0010715303406543 | full_train |
| knn_k_5_core_without_lag | models | KNN | core_without_lag | 36.87710050810092 | 1547.0786139852162 | 0.0007831441533865 | resource_limited_15000_train_rows |
| knn_k_15_core_without_lag | models | KNN | core_without_lag | 37.41297413492964 | 1547.4694968052554 | 0.0002781587696996 | resource_limited_15000_train_rows |

## Warnings

| model_run_id | warning_summary |
| --- | --- |
| lasso_alpha_0_0001_core_without_lag | Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.306e+07, tolerance: 1.764e+04 |
| lasso_alpha_0_001_core_without_lag | Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.877e+04, tolerance: 1.764e+04 |
| linearsvr_c_10_core_without_lag | Liblinear failed to converge, increase the number of iterations. |

## Failed Runs

None.

The 2013-2014 test split has not been used for preprocessing, feature selection, hyperparameter tuning, model selection or final evaluation.
