"""
config.py
=========
Central configuration for the Demystifying Credit Risk pipeline.
Every seed, path, threshold, and cost weight used anywhere in the codebase
is defined here once, reproducibility claim is literally
true: change nothing, run run_pipeline.py, get the same numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Kaggle dataset identifier
KAGGLE_DATASET = "wordsforthewise/lending-club"

# Raw file locations in data/
ACCEPTED_FILENAME = "accepted_2007_to_2018Q4.csv"
REJECTED_FILENAME = "rejected_2007_to_2018Q4.csv"

ACCEPTED_PATH = DATA_DIR / ACCEPTED_FILENAME
REJECTED_PATH = DATA_DIR / REJECTED_FILENAME

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
APPROVAL_THRESHOLD = 0.5   # tau

MODEL_PARAMS = {
    "random_forest": dict(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
    "xgboost": dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                     eval_metric="logloss", n_jobs=-1),
    "mlp": dict(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
                random_state=SEED, early_stopping=True),
}

# ---------------------------------------------------------------------------
# FintechPrescriptiveRecourse (FPR)
# Each entry: feature -> (direction, step_size, cost_weight, max_steps)
# direction: -1 means "decrease reduces risk", +1 means "increase reduces risk"
# ---------------------------------------------------------------------------
FPR_MAX_ITERATIONS = 12

FPR_ACTIONABLE_FEATURES_STAGE_B = {
    "revol_util": (-1, 5.0, 1.0, 6),
    "dti": (-1, 1.0, 1.0, 8),
    "term_months": (-1, 24.0, 1.5, 2),
    "loan_amnt": (-1, 1000.0, 2.5, 10),
}

FPR_ACTIONABLE_FEATURES_STAGE_A = {
    "dti": (-1, 1.0, 1.0, 8),
    "loan_amnt_requested": (-1, 1000.0, 2.5, 10),
}