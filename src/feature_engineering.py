"""
feature_engineering.py
=======================
Domain-specific feature engineering and the scikit-learn ColumnTransformer
preprocessing pipelines for Stage B (default prediction) and Stage A
(acceptance prediction). See Chapter Three, Section 3.5.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# Stage B: accepted-loan feature engineering
# ---------------------------------------------------------------------------
# "Ideal" feature set for the full Kaggle release (151 fields). Every entry
# here is only used if actually present in the incoming dataframe, so the
# same code degrades gracefully to the smaller H2O.ai fallback schema
# without special-casing -- this is what "taking the full dataset's ideal
# columns into account" means concretely: a superset list, filtered at
# runtime, not a hard-coded assumption about which schema is in play.
STAGE_B_IDEAL_NUMERIC = [
    "loan_amnt", "int_rate", "installment", "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "revol_util", "revol_bal", "total_acc", "open_acc",
    "pub_rec", "pub_rec_bankruptcies", "mort_acc", "longest_credit_length",
    "term_months", "grade_ordinal", "fico_avg",
    "loan_to_income", "installment_burden", "revol_delinq_interaction",
    "credit_velocity", "log_annual_inc", "addr_state_freq",
]
STAGE_B_IDEAL_CATEGORICAL = [
    "home_ownership", "purpose", "verification_status", "application_type",
]
STAGE_B_TARGET = "is_default"


def _select_available(df: pd.DataFrame, candidates: list) -> list:
    return [c for c in candidates if c in df.columns]


def engineer_stage_b_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["installment_burden"] = df["loan_amnt"] / df["term_months"]
    df["revol_delinq_interaction"] = df["revol_util"] * (df["delinq_2yrs"] + 1)
    df["credit_velocity"] = df["total_acc"] / (df["longest_credit_length"] + 1)
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    state_freq = df["addr_state"].value_counts(normalize=True)
    df["addr_state_freq"] = df["addr_state"].map(state_freq)
    return df


def get_stage_b_feature_lists(df: pd.DataFrame) -> tuple[list, list]:
    """Returns (numeric_features, categorical_features) actually available
    in `df`, drawn from the ideal superset above. Call this *after*
    engineer_stage_b_features() so the engineered columns are present to
    be selected."""
    numeric = _select_available(df, STAGE_B_IDEAL_NUMERIC)
    categorical = _select_available(df, STAGE_B_IDEAL_CATEGORICAL)
    return numeric, categorical


def build_stage_b_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])


# ---------------------------------------------------------------------------
# Stage A: acceptance/rejection feature engineering (shared 6-field subset)
# ---------------------------------------------------------------------------
STAGE_A_NUMERIC = ["loan_amnt_requested", "dti", "emp_length", "risk_score", "zip3_freq"]
STAGE_A_CATEGORICAL = ["state"]
STAGE_A_TARGET = "accepted"  # 1 = accepted (funded), 0 = rejected


def engineer_stage_a_features(accepted_df: pd.DataFrame,
                               rejected_df: pd.DataFrame) -> pd.DataFrame:
    """Build the combined Stage A dataset from the accepted and rejected
    populations, restricted to their six shared fields (Chapter Three,
    Section 3.2). `accepted_df` must already carry a `loan_amnt_requested`
    equivalent (falls back to `loan_amnt`, i.e. amount funded, if the
    original requested amount is not separately available) and a
    `risk_score` equivalent (falls back to the mean of fico_range_low/high
    if present, since accepted-side data reports FICO as a range).
    """
    acc = accepted_df.copy()
    if "loan_amnt_requested" not in acc.columns:
        acc["loan_amnt_requested"] = acc.get("loan_amnt")
    if "risk_score" not in acc.columns:
        if {"fico_range_low", "fico_range_high"}.issubset(acc.columns):
            acc["risk_score"] = (acc["fico_range_low"] + acc["fico_range_high"]) / 2
        else:
            acc["risk_score"] = np.nan  # will be median-imputed downstream
    acc["accepted"] = 1

    rej = rejected_df.copy()
    rej["accepted"] = 0

    shared_cols = ["loan_amnt_requested", "dti", "emp_length", "risk_score",
                    "state", "accepted"]
    if "zip3" in acc.columns:
        shared_cols.insert(-1, "zip3")
    elif "zip_code" in acc.columns:
        acc = acc.rename(columns={"zip_code": "zip3"})
        shared_cols.insert(-1, "zip3")
    if "zip3" in rej.columns:
        pass
    else:
        rej["zip3"] = np.nan

    for c in shared_cols:
        if c not in acc.columns:
            acc[c] = np.nan
        if c not in rej.columns:
            rej[c] = np.nan

    combined = pd.concat([acc[shared_cols], rej[shared_cols]], ignore_index=True)

    zip3_freq = combined["zip3"].value_counts(normalize=True)
    combined["zip3_freq"] = combined["zip3"].map(zip3_freq).fillna(0.0)
    combined = combined.drop(columns=["zip3"])

    for c in ("loan_amnt_requested", "dti", "emp_length", "risk_score"):
        if combined[c].isna().sum() > 0:
            combined[c] = combined[c].fillna(combined[c].median())

    return combined


def build_stage_a_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), STAGE_A_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE_A_CATEGORICAL),
    ])
