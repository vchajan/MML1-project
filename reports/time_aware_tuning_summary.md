# Time-Aware Model Tuning Summary

## Time-CV folds
| fold_id | fit_start_year | fit_end_year | evaluation_start_year | evaluation_end_year |
| --- | --- | --- | --- | --- |
| fold_1 | 1997 | 2004 | 2005 | 2006 |
| fold_2 | 1997 | 2006 | 2007 | 2008 |
| fold_3 | 1997 | 2008 | 2009 | 2010 |

## Rolling lag assumption
Previous-year official yield is assumed to be available when predicting the following year.

## Baseline CV results
| model_run_id | mean_mae | std_mae | mean_rmse | worst_fold_mae |
| --- | --- | --- | --- | --- |
| cv_baseline_lag_with_crop_median_fallback | 1.0748767127096217 | 0.033504014848858316 | 4.377610351646491 | 1.1219907041916857 |
| cv_baseline_crop_median | 2.163593996929167 | 0.02482183032966197 | 6.9080871137429005 | 2.198631278287617 |
| cv_baseline_global_median | 3.963961589534763 | 0.047262440473364364 | 12.92147449227278 | 4.019161875713578 |

## Direct model CV results
| model_run_id | model_name | feature_set | mean_mae | std_mae | worst_fold_mae |
| --- | --- | --- | --- | --- | --- |
| cv_direct_rf_200_depth_none_leaf_20_maxfeat_0_5_core_with_lag | RandomForest | core_with_lag | 1.259570264364367 | 0.0389547310518315 | 1.29568393324813 |
| cv_direct_linearsvr_c_0_03_epsilon_0_0_core_with_lag | LinearSVR | core_with_lag | 1.287105254173979 | 0.0408839064675475 | 1.3376254720909244 |
| cv_direct_linearsvr_c_0_03_epsilon_0_1_core_with_lag | LinearSVR | core_with_lag | 1.2892554351095578 | 0.0401273379533795 | 1.3387812654841833 |
| cv_direct_linearsvr_c_0_1_epsilon_0_0_core_with_lag | LinearSVR | core_with_lag | 1.3160312846655 | 0.0440043227073908 | 1.3710288903574772 |
| cv_direct_linearsvr_c_0_1_epsilon_0_1_core_with_lag | LinearSVR | core_with_lag | 1.3178148330421626 | 0.0426708165999481 | 1.3713883227547583 |
| cv_direct_linearsvr_c_0_3_epsilon_0_0_core_with_lag | LinearSVR | core_with_lag | 1.3360740460703229 | 0.0490384317781213 | 1.399438609193347 |
| cv_direct_linearsvr_c_0_3_epsilon_0_1_core_with_lag | LinearSVR | core_with_lag | 1.3378696986668397 | 0.0492186230897395 | 1.401612949634881 |
| cv_direct_linearsvr_c_1_0_epsilon_0_0_core_with_lag | LinearSVR | core_with_lag | 1.344482889267019 | 0.0529581361165773 | 1.4137893248637106 |
| cv_direct_linearsvr_c_1_0_epsilon_0_1_core_with_lag | LinearSVR | core_with_lag | 1.3465729097426689 | 0.052905126961623 | 1.4156240653926877 |
| cv_direct_tree_depth_none_leaf_50_core_with_lag | DecisionTree | core_with_lag | 1.4175763991111232 | 0.087713630256196 | 1.5386289996022744 |
| cv_direct_tree_depth_16_leaf_10_core_without_lag | DecisionTree | core_without_lag | 1.4525952726565634 | 0.0465476225776532 | 1.5042886416218406 |
| cv_direct_tree_depth_none_leaf_20_core_without_lag | DecisionTree | core_without_lag | 1.4660747882134686 | 0.0420147286227525 | 1.5163337210261052 |

## Residual model CV results
| model_run_id | model_name | mean_mae | std_mae | worst_fold_mae |
| --- | --- | --- | --- | --- |
| cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector | LinearSVR | 1.083048732898592 | 0.0344348320497914 | 1.126521812374789 |
| cv_residual_linearsvr_c_0_1_epsilon_0_0_residual_lag_corrector | LinearSVR | 1.0881625118176383 | 0.0347666969073015 | 1.1286232159507994 |
| cv_residual_linearsvr_c_0_3_epsilon_0_0_residual_lag_corrector | LinearSVR | 1.0910973267470512 | 0.0350772416686347 | 1.129396863959644 |
| cv_residual_linearsvr_c_1_0_epsilon_0_0_residual_lag_corrector | LinearSVR | 1.0922925897011826 | 0.0352243865628721 | 1.1296564526234492 |
| cv_residual_rf_250_depth_none_leaf_20_maxfeat_sqrt_residual_lag_corrector | RandomForest | 1.1271713728622348 | 0.0202223159643459 | 1.1457887860290934 |
| cv_residual_rf_200_depth_20_leaf_10_maxfeat_sqrt_residual_lag_corrector | RandomForest | 1.1293927374923949 | 0.0207041195290775 | 1.1520320764920466 |
| cv_residual_ridge_alpha_100_0_residual_lag_corrector | Ridge | 2.288013268992702 | 0.0331236756466364 | 2.3347145441253883 |
| cv_residual_ridge_alpha_10_0_residual_lag_corrector | Ridge | 2.3104019650695435 | 0.0241139807546191 | 2.34394820894162 |
| cv_residual_ridge_alpha_1_0_residual_lag_corrector | Ridge | 2.323212987792166 | 0.0247178892873764 | 2.357471203188997 |
| cv_residual_ridge_alpha_0_1_residual_lag_corrector | Ridge | 2.325189560639205 | 0.0245999567506104 | 2.359408466793931 |

## Log-target model CV results
| model_run_id | model_name | mean_mae | std_mae | worst_fold_mae |
| --- | --- | --- | --- | --- |
| cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag | RandomForest | 1.578011845629813 | 0.0665591142212535 | 1.672111594656472 |
| cv_log_target_rf_250_depth_none_leaf_20_maxfeat_sqrt_core_with_lag | RandomForest | 1.677222995876286 | 0.0681007882172206 | 1.7735109700904348 |
| cv_log_target_linearsvr_c_0_3_epsilon_0_1_core_with_lag | LinearSVR | 1.765174642734781 | 0.1504875126375975 | 1.9709680526133424 |
| cv_log_target_linearsvr_c_0_1_epsilon_0_1_core_with_lag | LinearSVR | 1.780532017564984 | 0.1624789694085537 | 2.002646525771951 |
| cv_log_target_linearsvr_c_0_3_epsilon_0_0_core_with_lag | LinearSVR | 1.7895619617775278 | 0.1761097371712159 | 2.0335011023385188 |
| cv_log_target_ridge_alpha_1_0_core_with_lag | Ridge | 1.792375491439041 | 0.0362800838038529 | 1.8285614573082047 |
| cv_log_target_ridge_alpha_10_0_core_with_lag | Ridge | 1.7981133385397543 | 0.0362364326041072 | 1.83574198442265 |
| cv_log_target_linearsvr_c_0_1_epsilon_0_0_core_with_lag | LinearSVR | 1.8076212348192129 | 0.188345209905562 | 2.0681571162819528 |

## Selected time-CV shortlist
| application_track | model_run_id | experiment_family | feature_set | time_cv_mean_mae | time_cv_worst_fold_mae | selection_reason |
| --- | --- | --- | --- | --- | --- | --- |
| forecast_with_lag | cv_baseline_lag_with_crop_median_fallback | baseline | lag_with_crop_fallback | 1.0748767127096217 | 1.1219907041916857 | forecast baseline using previous-year yield with crop median fallback |
| forecast_with_lag | cv_direct_rf_200_depth_none_leaf_20_maxfeat_0_5_core_with_lag | direct | core_with_lag | 1.259570264364367 | 1.29568393324813 | top direct time-CV model with core_with_lag |
| forecast_with_lag | cv_direct_linearsvr_c_0_03_epsilon_0_0_core_with_lag | direct | core_with_lag | 1.287105254173979 | 1.3376254720909244 | top direct time-CV model with core_with_lag |
| forecast_with_lag | cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector | residual | residual_lag_corrector | 1.083048732898592 | 1.126521812374789 | top residual time-CV model correcting lag baseline |
| forecast_with_lag | cv_residual_linearsvr_c_0_1_epsilon_0_0_residual_lag_corrector | residual | residual_lag_corrector | 1.0881625118176383 | 1.1286232159507994 | top residual time-CV model correcting lag baseline |
| forecast_with_lag | cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag | log_target | core_with_lag | 1.5780118456298127 | 1.672111594656472 | top stable log-target time-CV model with core_with_lag |
| suitability_without_lag | cv_baseline_crop_median | baseline | crop_only | 2.163593996929167 | 2.198631278287617 | suitability baseline without previous-year yield |
| suitability_without_lag | cv_direct_tree_depth_16_leaf_10_core_without_lag | direct | core_without_lag | 1.4525952726565634 | 1.5042886416218406 | top direct time-CV model ranked only within core_without_lag |
| suitability_without_lag | cv_direct_tree_depth_none_leaf_20_core_without_lag | direct | core_without_lag | 1.4660747882134686 | 1.5163337210261052 | top direct time-CV model ranked only within core_without_lag |

## Validation shortlist results
| application_track | model_run_id | experiment_family | model_name | feature_set | validation_mae | validation_rmse | validation_r2 | time_cv_mean_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_with_lag | cv_baseline_lag_with_crop_median_fallback | baseline | LagWithCropMedianFallback | lag_with_crop_fallback | 1.233652348338793 | 5.517610430294291 | 0.8081722125503343 | 1.0748767127096217 |
| forecast_with_lag | cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector | residual | LinearSVR | residual_lag_corrector | 1.2408891719969837 | 5.42449574785008 | 0.8145921166543428 | 1.083048732898592 |
| forecast_with_lag | cv_residual_linearsvr_c_0_1_epsilon_0_0_residual_lag_corrector | residual | LinearSVR | residual_lag_corrector | 1.2418372106606057 | 5.419428449001443 | 0.8149383528288927 | 1.0881625118176383 |
| forecast_with_lag | cv_direct_rf_200_depth_none_leaf_20_maxfeat_0_5_core_with_lag | direct | RandomForest | core_with_lag | 1.3666005417112064 | 5.434991601356033 | 0.813873931317448 | 1.259570264364367 |
| forecast_with_lag | cv_direct_linearsvr_c_0_03_epsilon_0_0_core_with_lag | direct | LinearSVR | core_with_lag | 1.515658681682611 | 6.205240926027085 | 0.7573799174902099 | 1.287105254173979 |
| suitability_without_lag | cv_direct_tree_depth_none_leaf_20_core_without_lag | direct | DecisionTree | core_without_lag | 1.6734770052892585 | 6.229360367292113 | 0.7554901492006221 | 1.4660747882134686 |
| suitability_without_lag | cv_direct_tree_depth_16_leaf_10_core_without_lag | direct | DecisionTree | core_without_lag | 1.727556502079843 | 6.961145429101619 | 0.6946690256426249 | 1.4525952726565634 |
| forecast_with_lag | cv_log_target_rf_200_depth_20_leaf_10_maxfeat_sqrt_core_with_lag | log_target | RandomForest | core_with_lag | 1.7312954013407678 | 6.824416678380979 | 0.7065456706233955 | 1.578011845629813 |
| suitability_without_lag | cv_baseline_crop_median | baseline | CropMedian | crop_only | 2.3431433744742174 | 7.490626669069437 | 0.6464541410925199 | 2.163593996929167 |

## Original validation benchmark comparison
- Original best baseline: baseline_lag_with_crop_median_fallback MAE=1.2336508421343648 RMSE=5.517607959798213
- Original best trained model: linearsvr_c_0_1_core_with_lag MAE=1.5336555887879595 RMSE=6.086053761116267

## Selected frozen configuration
- Best forecast validation result overall: cv_baseline_lag_with_crop_median_fallback
- Best trained forecast validation model: cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector
- Best suitability validation model: cv_direct_tree_depth_none_leaf_20_core_without_lag
- Selected run: cv_residual_linearsvr_c_0_03_epsilon_0_0_residual_lag_corrector
- Residual beat lag baseline in mean CV MAE: false
- Log-target improved over direct in mean CV MAE: false

## Test usage
- Test 2013-2014 was not opened.
- test_data_accessed: false
- test_used_for_selection: false
