# Validation Benchmark Summary

- Train rows: 202166
- Validation rows: 32388
- Test data accessed: false
- Final test evaluation: not performed
- Selected feature set: `core_with_lag`
- Best model: `linearsvr_c_0_1_core_with_lag`
- Best baseline: `baseline_lag_with_crop_median_fallback`
- Absolute MAE improvement over baseline: -0.300005
- Relative MAE improvement over baseline: -24.32%

Overfitting cannot be determined definitively from validation results alone.
KNN uses resource-limited training with at most 15,000 train rows, so it is not fully directly comparable to full-train models.

## Baselines

| model_run_id | mae | rmse | r2 | median_ae |
| --- | --- | --- | --- | --- |
| baseline_global_median | 4.096403565530293 | 13.134072227001065 | -0.0869467046432839 | 0.6332223165992091 |
| baseline_crop_median | 2.343140949468887 | 7.490617339952081 | 0.6464550217314571 | 0.4335613552671066 |
| baseline_lag_with_crop_median_fallback | 1.2336508421343648 | 5.517607959798213 | 0.8081723843311066 | 0.1815815060592565 |

## Feature Set Comparison

| model_run_id | feature_set | model_name | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| feature_set_ridge_core_without_lag | core_without_lag | Ridge | 3.213704257920371 | 7.203451331881567 | 0.6730429529091202 |
| feature_set_decision_tree_core_without_lag | core_without_lag | DecisionTree | 1.7787435618966576 | 8.819141411035464 | 0.5099255485227814 |
| feature_set_ridge_core_with_lag | core_with_lag | Ridge | 3.1420268443768578 | 7.040696685952627 | 0.6876505681888669 |
| feature_set_decision_tree_core_with_lag | core_with_lag | DecisionTree | 1.661591384096158 | 9.94986223310044 | 0.3762025648706997 |

## All Successful Models

| model_run_id | phase | model_name | feature_set | mae | rmse | r2 | training_scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_lag_with_crop_median_fallback | baselines | LagWithCropMedianFallback | lag_with_crop_fallback | 1.2336508421343648 | 5.517607959798213 | 0.8081723843311066 | full_train |
| baseline_crop_median | baselines | CropMedian | crop_only | 2.343140949468887 | 7.490617339952081 | 0.6464550217314571 | full_train |
| baseline_global_median | baselines | GlobalMedian | none | 4.096403565530293 | 13.134072227001065 | -0.0869467046432839 | full_train |
| feature_set_decision_tree_core_with_lag | feature-sets | DecisionTree | core_with_lag | 1.661591384096158 | 9.94986223310044 | 0.3762025648706997 | full_train |
| feature_set_decision_tree_core_without_lag | feature-sets | DecisionTree | core_without_lag | 1.7787435618966576 | 8.819141411035464 | 0.5099255485227814 | full_train |
| feature_set_ridge_core_with_lag | feature-sets | Ridge | core_with_lag | 3.1420268443768578 | 7.040696685952627 | 0.6876505681888669 | full_train |
| feature_set_ridge_core_without_lag | feature-sets | Ridge | core_without_lag | 3.213704257920371 | 7.203451331881567 | 0.6730429529091202 | full_train |
| linearsvr_c_0_1_core_with_lag | models | LinearSVR | core_with_lag | 1.5336555887879595 | 6.086053761116267 | 0.7666106585908892 | full_train |
| linearsvr_c_1_core_with_lag | models | LinearSVR | core_with_lag | 1.5360445328739878 | 6.043547269173161 | 0.7698593707215335 | full_train |
| linearsvr_c_10_core_with_lag | models | LinearSVR | core_with_lag | 1.5367778406559751 | 6.034554505795134 | 0.7705437570112678 | full_train |
| tree_depth_none_leaf_20_core_with_lag | models | DecisionTree | core_with_lag | 1.5674966260674164 | 6.981063708920688 | 0.6929192076738615 | full_train |
| tree_depth_15_leaf_10_core_with_lag | models | DecisionTree | core_with_lag | 1.661591384096158 | 9.94986223310044 | 0.3762025648706997 | full_train |
| rf_150_depth_20_leaf_5_core_with_lag | models | RandomForest | core_with_lag | 1.6664809758020858 | 5.766764079362387 | 0.7904566807807907 | full_train |
| rf_200_depth_none_leaf_10_core_with_lag | models | RandomForest | core_with_lag | 1.6830544946089594 | 5.817350213491195 | 0.7867643230151325 | full_train |
| tree_depth_8_leaf_20_core_with_lag | models | DecisionTree | core_with_lag | 1.710804056922431 | 7.143238540698304 | 0.6784860981101672 | full_train |
| lasso_alpha_0_001_core_with_lag | models | Lasso | core_with_lag | 3.056148000595099 | 7.0213496205100325 | 0.6893648139102019 | full_train |
| ridge_alpha_10_core_with_lag | models | Ridge | core_with_lag | 3.1188305370196208 | 7.0314355940566 | 0.6884717353100618 | full_train |
| lasso_alpha_0_0001_core_with_lag | models | Lasso | core_with_lag | 3.1286979580405165 | 7.036994070798712 | 0.687979003198651 | full_train |
| ridge_alpha_1_core_with_lag | models | Ridge | core_with_lag | 3.1420268443768578 | 7.040696685952627 | 0.6876505681888669 | full_train |
| ridge_alpha_0_1_core_with_lag | models | Ridge | core_with_lag | 3.1454993741257047 | 7.040842599901134 | 0.6876376215691755 | full_train |
| linear_regression_core_with_lag | models | LinearRegression | core_with_lag | 3.147977721089253 | 7.0412830140582106 | 0.687598542974759 | full_train |
| knn_k_5_core_with_lag | models | KNN | core_with_lag | 3.397576581466384 | 11.912072256996248 | 0.1059041316693898 | resource_limited_15000_train_rows |
| knn_k_15_core_with_lag | models | KNN | core_with_lag | 3.970384351269074 | 10.709365790049045 | 0.2773351107602793 | resource_limited_15000_train_rows |

## Warnings

| model_run_id | warning_summary |
| --- | --- |
| lasso_alpha_0_0001_core_with_lag | Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.158e+07, tolerance: 1.763e+04 |
| lasso_alpha_0_001_core_with_lag | Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.898e+04, tolerance: 1.763e+04 |
| linearsvr_c_10_core_with_lag | Liblinear failed to converge, increase the number of iterations. |

## Failed Runs

None.

The 2013-2014 test split has not been used for preprocessing, feature selection, hyperparameter tuning, model selection or final evaluation.
