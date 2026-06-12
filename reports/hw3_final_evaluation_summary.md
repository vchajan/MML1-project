# HW3 – Final frozen test evaluation

## Experimental protocol

- model list, features, preprocessing and hyperparameters were frozen before
  the final test file was restored,
- model-selection period: train 1997–2010 and validation 2011–2012,
- final fit period: 1997–2012,
- final test period: 2013–2014,
- common feature set: `core_without_lag`,
- primary metric: MAE,
- final test metrics were not used for further tuning.

Frozen-plan SHA-256:

`1b4209c8605e0fca704c32d1d6291deb4f3134c2b007ec2db7d6be6bce793249`

## Pre-selected model

Validation selected **Random Forest**
(`random_forest_200_leaf_20`).

Validation metrics:

- MAE: 1.618119,
- RMSE: 5.712142,
- R²: 0.794407.

Final test metrics:

- MAE: 1.611674,
- RMSE: 5.277702,
- R²: 0.839797,
- MedianAE: 0.424710.

Improvement against the mean baseline:

- absolute MAE: 4.327191,
- relative MAE: 72.86 %.

Improvement against the median baseline:

- absolute MAE: 2.907632,
- relative MAE: 64.34 %.

## Test ranking

The test ranking is reported for comparison only. It does not change the
pre-selected model or any configuration.

Test-best run: **Random Forest**
(`random_forest_200_leaf_20`), test MAE
1.611674.

| model_name              | test_mae | test_rmse | test_r2   | mae_generalization_gap |
| ----------------------- | -------- | --------- | --------- | ---------------------- |
| Random Forest           | 1.611674 | 5.277702  | 0.839797  | 0.481418               |
| Decision Tree           | 1.612646 | 6.138621  | 0.783268  | 0.660778               |
| LinearSVR               | 2.505427 | 7.923610  | 0.638901  | 0.297459               |
| Lasso                   | 2.946074 | 7.185825  | 0.703015  | 0.087973               |
| Gradient Boosting       | 2.951746 | 7.400100  | 0.685040  | 0.206849               |
| Elastic Net             | 3.065463 | 7.574680  | 0.670004  | 0.076393               |
| Ridge                   | 3.154957 | 7.071487  | 0.712391  | 0.108059               |
| Linear Regression       | 3.167277 | 7.076283  | 0.712001  | 0.122653               |
| DummyRegressor (median) | 4.519306 | 13.827940 | -0.099754 | 0.507010               |
| DummyRegressor (mean)   | 5.938865 | 13.200952 | -0.002285 | 0.110458               |

## Validation-to-test stability

| model_name              | validation_mae | test_mae | test_minus_validation_mae | validation_rmse | test_rmse | validation_r2 | test_r2   |
| ----------------------- | -------------- | -------- | ------------------------- | --------------- | --------- | ------------- | --------- |
| Random Forest           | 1.618119       | 1.611674 | -0.006446                 | 5.712142        | 5.277702  | 0.794407      | 0.839797  |
| Decision Tree           | 1.673477       | 1.612646 | -0.060831                 | 6.229360        | 6.138621  | 0.755490      | 0.783268  |
| LinearSVR               | 2.339304       | 2.505427 | 0.166123                  | 7.752574        | 7.923610  | 0.621295      | 0.638901  |
| Gradient Boosting       | 2.761550       | 2.951746 | 0.190196                  | 7.277977        | 7.400100  | 0.666243      | 0.685040  |
| Lasso                   | 2.979749       | 2.946074 | -0.033675                 | 7.249110        | 7.185825  | 0.668885      | 0.703015  |
| Elastic Net             | 3.021564       | 3.065463 | 0.043899                  | 7.515958        | 7.574680  | 0.644059      | 0.670004  |
| Ridge                   | 3.175822       | 3.154957 | -0.020865                 | 7.122269        | 7.071487  | 0.680371      | 0.712391  |
| Linear Regression       | 3.222497       | 3.167277 | -0.055221                 | 7.136064        | 7.076283  | 0.679132      | 0.712001  |
| DummyRegressor (median) | 4.096406       | 4.519306 | 0.422900                  | 13.134077       | 13.827940 | -0.086947     | -0.099754 |
| DummyRegressor (mean)   | 5.639314       | 5.938865 | 0.299551                  | 12.599494       | 13.200952 | -0.000266     | -0.002285 |

A train–test or validation–test gap is not automatically proof of
overfitting. Because the split is chronological, the difference can also
reflect temporal distribution shift between historical and later years.

## Residual analysis of the validation-selected Random Forest

Positive residual means underprediction; negative residual means
overprediction.

| target_quartile | rows | actual_min | actual_max | mae      | rmse      | mean_residual_actual_minus_prediction | underprediction_rate_pct |
| --------------- | ---- | ---------- | ---------- | -------- | --------- | ------------------------------------- | ------------------------ |
| Q1 lowest       | 8151 | 0.000000   | 0.657143   | 0.561684 | 1.264081  | -0.547826                             | 9.716599                 |
| Q2              | 8147 | 0.657187   | 1.183097   | 0.495168 | 1.035650  | -0.419643                             | 25.935927                |
| Q3              | 8195 | 1.183562   | 3.000000   | 0.678079 | 1.397594  | -0.270822                             | 50.262355                |
| Q4 highest      | 8103 | 3.001194   | 362.000000 | 4.734646 | 10.362811 | 2.884718                              | 73.590028                |

## Target distribution comparison

| split                 | rows   | mean     | std       | min      | p50      | p90       | p95       | p99       | max         |
| --------------------- | ------ | -------- | --------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| train_1997_2010       | 202164 | 4.505971 | 19.356300 | 0.000000 | 0.996843 | 8.829255  | 19.790651 | 68.000000 | 2247.533300 |
| validation_2011_2012  | 32388  | 4.711542 | 12.598011 | 0.000000 | 1.137926 | 9.533607  | 22.081163 | 70.413000 | 311.000000  |
| final_train_1997_2012 | 234552 | 4.534358 | 18.570146 | 0.000000 | 1.000000 | 8.981842  | 20.000000 | 68.400000 | 2247.533300 |
| test_2013_2014        | 32596  | 5.164614 | 13.186101 | 0.000000 | 1.183329 | 11.304705 | 23.390930 | 73.733333 | 362.000000  |

## Resource-limited experiment

KNN and RBF-SVR remain in a separate deterministic sample experiment because
their costs scale poorly on the full high-dimensional dataset. Their metrics
must not be mixed directly with the full-data ranking.

| model_name                     | test_mae | test_rmse | test_r2   |
| ------------------------------ | -------- | --------- | --------- |
| Decision Tree (sample)         | 2.646002 | 7.027429  | 0.710185  |
| KNN (sample)                   | 4.090619 | 9.865669  | 0.428810  |
| DummyRegressor median (sample) | 4.506452 | 13.702038 | -0.101788 |
| SVR RBF (sample)               | 5.489616 | 9.155102  | 0.508126  |
| DummyRegressor mean (sample)   | 5.839574 | 13.076974 | -0.003557 |

## Interpretation limits

The models predict crop yield, not economic profit and not a causal effect.
The dataset does not include complete costs of labour, irrigation,
fertiliser, pesticides, transport or crop selling prices.
