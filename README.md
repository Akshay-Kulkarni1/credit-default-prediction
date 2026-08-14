# Credit Default Prediction

Predicting credit card default from monthly customer statement data. The pipeline
engineers customer-level features from a panel of statements, selects features with
XGBoost, tunes a gradient-boosted classifier, explains it with SHAP, and translates
the predicted probability of default into an acceptance-threshold strategy. A neural
network grid search is included as a benchmark.

## Repository structure

```
credit-default-prediction/
├── data/                          # Input data and generated artifacts (gitignored)
├── src/
│   └── credit_default_pipeline.py # End-to-end pipeline
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/credit-default-prediction.git
cd credit-default-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place `sampled_train_data.csv` in `data/` before running.

## Usage

```bash
python src/credit_default_pipeline.py
```

`DATA_DIR` defaults to `data/` alongside the repository, resolved relative to the
script rather than the working directory. Point it elsewhere with an environment
variable:

```bash
export ML_PROJECT_DATA_DIR=/path/to/data     # Windows: set ML_PROJECT_DATA_DIR=...
```

## Pipeline

| Step | Stage | Output |
| --- | --- | --- |
| 1–3 | Load statements, one-hot encode categoricals, filter to April 2018 | — |
| 4 | Aggregation features: mean, sum, min, max, std per customer | — |
| 5 | Rolling features over 3/6/9/12-month windows | — |
| 6 | Recency: days since last statement | — |
| 7 | Delinquency-risk ratio (`D_avg` / `R_avg`) | — |
| 8 | Response-rate and ever-response features per window | — |
| 9 | Train / test1 / test2 split (70 / 15 / 15) | — |
| 10 | Feature selection from two XGBoost models, importance > 0.005 | `feature_importance_1.csv`, `feature_importance_2.csv` |
| 11–12 | Grid search over 72 configurations, stability check | `grid_search_results.csv` |
| 13 | Final XGBoost model | `xgb_model.pkl` |
| 14–15 | SHAP summary statistics, beeswarm and waterfall plots | `Shap_Analysis.csv` |
| 16 | Acceptance-threshold strategy: default rate and expected revenue | `train_df_PD.csv` |
| 17–18 | Neural network grid search and stability check | `nn_grid_results.csv` |

Reference date for all time-windowed features is 2018-03-31.

Expected revenue assumes an annual horizon at 2% of average balance plus 0.1% of
average spend per month, with revenue set to zero for defaulters. Aggressive and
conservative acceptance thresholds were evaluated at 0.93 and 0.6.

## Dependencies

`xgboost` is pinned below 2.0 because the classifiers pass `use_label_encoder`,
which was removed in 2.0. `numpy` is pinned below 2.0 for compatibility with that
XGBoost range and with `shap`.
