# HW3 – Pre-test model selection summary

## Guardrails

- train: 1997–2010,
- validation: 2011–2012,
- common feature set: `core_without_lag`,
- test file absent during all pre-test runs,
- full 1997–2014 model dataset absent during all pre-test runs,
- no neural networks,
- model list and hyperparameters frozen before final test evaluation.

## Full-data validation result

The lowest validation MAE was achieved by **Random Forest**
(`random_forest_200_leaf_20`):

- validation MAE: 1.618119,
- validation RMSE: 5.712142,
- validation R²: 0.794407.

Improvement against the mean baseline:

- absolute MAE improvement: 4.021194,
- relative MAE improvement: 71.31 %.

Improvement against the median baseline:

- absolute MAE improvement: 2.478286,
- relative MAE improvement: 60.50 %.

## Resource-limited validation experiment

KNN and RBF SVR were evaluated separately on a deterministic sample because
their prediction/training costs scale poorly for the full high-dimensional
one-hot encoded dataset.

Resource-limited validation order:

| model_name                     | validation_mae | validation_rmse | validation_r2 |
| ------------------------------ | -------------- | --------------- | ------------- |
| Decision Tree (sample)         | 2.617752       | 6.778063        | 0.663265      |
| KNN (sample)                   | 3.521587       | 10.010783       | 0.265464      |
| DummyRegressor median (sample) | 3.802955       | 12.169615       | -0.085502     |
| DummyRegressor mean (sample)   | 5.052218       | 11.687132       | -0.001136     |
| SVR RBF (sample)               | 5.156412       | 9.174670        | 0.383039      |

## Frozen decision

All configurations listed in
`data/reference/hw3_frozen_evaluation_plan.json` are frozen for the final
2013–2014 evaluation. Validation metrics may be discussed for model selection.
Final test metrics must not be used for further tuning or changing the model
list.
