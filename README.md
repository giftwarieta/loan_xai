# Demystifying Credit Risk
### An Explainable AI (XAI) Framework for Loan Default Prediction in Inclusive Lending

MSc Data Science thesis codebase. Two linked models on LendingClub data:

- **Stage A (acceptance model)** — trained on the combined accepted + rejected
  application population, restricted to the six fields both share, models
  *who gets rejected and why*.
- **Stage B (default model)** — trained on the accepted population alone,
  using its full feature set, models *who defaults after being accepted*.

Both stages share an identical explainability layer (SHAP + LIME) and both
feed into **FintechPrescriptiveRecourse (FPR)**, this thesis's original
contribution: a SHAP-guided, cost-weighted, actionability-constrained
algorithm that turns an explanation into a concrete action plan. See
Chapter Three, Section 3.9, for the full mathematical formalisation.

## Setup

```bash
pip install -r requirements.txt

# Optional but recommended: enables the real Kaggle data (both stages).
# Get a token from https://www.kaggle.com/settings -> API -> Create New Token
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Running

```bash
# 1. Sanity-check both data sources before committing to a full run
python quick_data_check.py

# 2. Explore the data interactively (charts render inline)
jupyter notebook notebooks/eda.ipynb

# 3. Run the full pipeline: cleaning -> feature engineering -> modelling
#    -> SHAP/LIME -> FintechPrescriptiveRecourse, for both stages
python run_pipeline.py
```

If `kaggle.json` is not configured, Stage B automatically falls back to a
smaller, real, publicly hosted LendingClub extract (163,987 accepted loans).
Stage A has no such fallback — it needs the real Kaggle
`rejected_2007_to_2018Q4.csv` — and `run_pipeline.py` will skip it cleanly
with a clear message rather than fabricate results.

## Project layout

```
LOAN_XAI/
├── data/                          # raw + cached CSVs (gitignored)
├── notebooks/
│   └── eda.ipynb                  # exploratory data analysis, both stages
├── outputs/
│   ├── figures/                   # confusion matrices, ROC curves, SHAP plots
│   ├── models/                    # fitted pipelines (.joblib)
│   └── reports/                   # metrics, SHAP/LIME summaries (.json/.csv)
├── src/
│   ├── config.py                  # seeds, paths, thresholds, FPR cost weights
│   ├── data_loader.py             # Kaggle-first, documented-fallback loading
│   ├── preprocessing.py           # cleaning for both stages
│   ├── feature_engineering.py     # engineered features + preprocessing pipelines
│   ├── models.py                  # Logistic Regression, Random Forest, XGBoost
│   ├── deep_learning.py           # MLP classifier
│   ├── evaluation.py              # metrics, confusion matrices, ROC curves
│   └── explainability.py          # SHAP + LIME
├── fintech_prescriptive_recourse.py  # the novel contribution (Ch.3, Sec.3.9)
├── run_pipeline.py                # orchestrates both stages end to end
├── quick_data_check.py             # fast pre-flight data sanity check
├── requirements.txt
└── README.md
```

## Reproducibility

Every stochastic operation — both train/test splits, every model's internal
random state, and every seeded sample drawn for SHAP, LIME, and the FPR
evaluation — is seeded from the single constant `config.SEED = 42`. Re-running
`run_pipeline.py` reproduces every number in Chapter Four exactly.
