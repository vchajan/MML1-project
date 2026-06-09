# Modeling Dataset Summary

- Input rows: 267150
- Output rows: 267148
- Train rows: 202164
- Validation rows: 32388
- Test rows: 32596
- Train years: 1997 to 2010
- Validation years: 2011 to 2012
- Test years: 2013 to 2014
- Feature columns without lag: 31
- Feature columns with lag: 33
- Categorical features: 4
- Numeric core features: 4
- Weather features: 23
- Rows with lag: 213184
- Rows without lag: 53964

## Lag Availability By Split

- train: 156366/202164 (0.7735)
- validation: 28001/32388 (0.8645)
- test: 28817/32596 (0.8841)

## Target Summary

- Minimum: 0.000000
- Maximum: 2247.533300
- Median: 1.000000
- Mean: 4.611258

## Train Target Quantiles

- 1%: 0.000000
- 5%: 0.170152
- 25%: 0.500000
- 50%: 0.996843
- 75%: 2.168739
- 95%: 19.790651
- 99%: 68.000000

## Unseen Categories

- Validation unseen categories: 19
- Test unseen categories: 43
- Unseen categories are retained and must be handled later with `OneHotEncoder(handle_unknown="ignore")` fitted only on train.

No preprocessing has been fitted during dataset construction.
The 2013-2014 test split has not been used for modeling, feature selection, preprocessing decisions, hyperparameter tuning, or model selection.

## Model Quality Exclusions

- Modeling rows before quality exclusion: 267150
- Modeling rows after quality exclusion: 267148
- Modeling-only excluded rows: 2
- Rows with changed lag availability/source after exclusion: 1
- Full canonical and model-base interim datasets are not modified by this script.
