"""
preprocessing.py
=================
Data cleaning for Stage B (accepted loans) and Stage A (accepted+rejected,
shared-field subset). See Chapter Three, Section 3.4, for the rationale
behind each choice (median imputation over dropping, percentile capping
over outright removal).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the binary default target from loan_status when the dataset
    does not already carry a pre-computed is_default/target field

    Follows standard practice in the literature reviewed in Chapter Two
    (e.g. Turiel & Aste, 2020): only loans with a matured, terminal status
    are kept. "Current", "In Grace Period", and "Late" loans have not yet
    reached a final outcome and are excluded rather than guessed at.
    """
    if "is_default" in df.columns:
        return df

    if "loan_status" not in df.columns:
        raise ValueError(
            "Neither 'is_default' nor 'loan_status' found -- cannot derive a "
            "target. Check the input schema."
        )

    bad_statuses = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
    good_statuses = {
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    }
    matured = df["loan_status"].isin(bad_statuses | good_statuses)
    n_excluded = int((~matured).sum())
    if n_excluded:
        logger.info(
            "Excluding %d loans with a non-terminal loan_status (Current, "
            "Late, In Grace Period) -- outcome not yet known.", n_excluded
        )
    df = df.loc[matured].copy()
    df["is_default"] = df["loan_status"].isin(bad_statuses).astype(int)
    return df


def _parse_emp_length(series: pd.Series) -> pd.Series:
    """Handles the real dataset's text encoding: '< 1 year' -> 0,
    '10+ years' -> 10, '3 years' -> 3, 'n/a' -> NaN."""
    s = series.astype(str).str.strip()
    s = s.replace({"n/a": np.nan, "nan": np.nan})
    out = s.str.extract(r"(\d+)")[0]
    out = pd.to_numeric(out, errors="coerce")
    out[s.str.contains("< 1", na=False)] = 0
    return out


def _derive_credit_history_length(df: pd.DataFrame) -> pd.DataFrame:
    """Real dataset has no pre-computed credit-history-length field; it is
    derived from issue_d minus earliest_cr_line (in years). The H2O.ai
    fallback already ships a pre-computed longest_credit_length column, in
    which case this is a no-op."""
    if "longest_credit_length" in df.columns:
        return df
    if {"issue_d", "earliest_cr_line"}.issubset(df.columns):
        issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        earliest_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["longest_credit_length"] = ((issue_dt - earliest_dt).dt.days / 365.25).round(2)
    return df


# Columns with near-total missingness in the full Kaggle release (hardship
# and settlement programme fields only populated for the small subset of
# borrowers who entered those programmes; joint-application fields only
# populated for joint loans). Dropped rather than imputed -- imputing a
# 99%-missing column manufactures information that was never observed.
_HIGH_MISSINGNESS_PREFIXES = (
    "hardship_", "settlement_", "sec_app_", "orig_projected_",
    "deferral_term", "payment_plan_start_date", "debt_settlement_flag_date",
)
_ID_AND_LEAKAGE_COLUMNS = (
    "id", "member_id", "url", "emp_title", "title", "desc",
    # post-origination / outcome-adjacent fields that would leak the
    # target if included as predictors
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "loan_status",
)


def clean_accepted_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the accepted-loan (Stage B) dataset. Returns the cleaned frame
    and a log dict recording every imputation/capping decision made, so
    Chapter Four's reported figures always trace back to this function.

    Handles both schemas transparently: the H2O.ai fallback mirror (15
    fields, pre-computed is_default/longest_credit_length) and the full
    Kaggle release (151 fields, loan_status text target, no pre-computed
    credit-history-length)."""
    log: dict = {}
    df = df.copy()

    drop_cols = [c for c in df.columns if c.startswith(_HIGH_MISSINGNESS_PREFIXES)]
    drop_cols += [c for c in _ID_AND_LEAKAGE_COLUMNS if c in df.columns and c != "loan_status"]
    log["columns_dropped_high_missingness_or_leakage"] = len(drop_cols)

    df = _derive_credit_history_length(df)
    df = _derive_target(df)  # must run before loan_status itself is dropped
    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)

    if "term" in df.columns:
        term_numeric = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        term_numeric = term_numeric.fillna(term_numeric.median())
        df["term_months"] = term_numeric.astype(int)
        df = df.drop(columns=["term"])

    if "emp_length" in df.columns:
        df["emp_length"] = _parse_emp_length(df["emp_length"])

    if "revol_util" in df.columns and not pd.api.types.is_numeric_dtype(df["revol_util"]):
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )

    if "grade" in df.columns:
        grade_map = {g: i for i, g in enumerate("ABCDEFG", start=1)}
        df["grade_ordinal"] = df["grade"].map(grade_map)
        df = df.drop(columns=["grade"])
    if "sub_grade" in df.columns:
        df = df.drop(columns=["sub_grade"])  # grade_ordinal captures the same signal more compactly

    if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df = df.drop(columns=["fico_range_low", "fico_range_high"])

    for c in ("home_ownership", "purpose", "addr_state", "verification_status",
              "application_type", "initial_list_status", "disbursement_method"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Impute any remaining numeric missingness with the median. Columns
    # with near-total missingness were already dropped above (or via the
    # explicit high-missingness prefix list), so what remains here is
    # genuinely worth keeping and imputing rather than discarding --
    # median imputation is applied regardless of the missingness fraction
    # for whatever survives to this point, since leaving any NaN in a
    # modelled numeric column would break StandardScaler downstream.
    numeric_candidates = df.select_dtypes(include=[np.number]).columns
    imputation_values = {}
    for c in numeric_candidates:
        if df[c].isna().sum() > 0:
            median_val = df[c].median()
            imputation_values[c] = float(median_val)
            df[c] = df[c].fillna(median_val)
    log["imputation_values"] = imputation_values

    for c in ("delinq_2yrs", "total_acc"):
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median()).astype(int)

    outlier_cols = [c for c in ("loan_amnt", "annual_inc", "dti", "revol_util")
                    if c in df.columns]
    outlier_bounds = {}
    for c in outlier_cols:
        lo, hi = df[c].quantile([0.01, 0.99])
        outlier_bounds[c] = {"lower": float(lo), "upper": float(hi)}
        df[c] = df[c].clip(lower=lo, upper=hi)
    log["outlier_bounds"] = outlier_bounds
    log["remaining_missing"] = int(df.isna().sum().sum())

    logger.info("Stage B cleaning complete: %s -> %s rows, %d columns",
                before, len(df), df.shape[1])
    return df, log


def clean_rejected_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the rejected-application dataset, restricted to fields that
    survive into the public record: Amount Requested, Application Date,
    Loan Title, Risk_Score, Debt-To-Income Ratio, Zip Code, State,
    Employment Length, Policy Code."""
    log: dict = {}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Amount Requested": "loan_amnt_requested",
        "Debt-To-Income Ratio": "dti",
        "Employment Length": "emp_length",
        "Risk_Score": "risk_score",
        "Zip Code": "zip3",
        "State": "state",
        "Application Date": "application_date",
        "Loan Title": "loan_title",
        "Policy Code": "policy_code",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "dti" in df.columns:
        df["dti"] = df["dti"].astype(str).str.replace("%", "", regex=False)
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    if "emp_length" in df.columns:
        extracted = df["emp_length"].astype(str).str.extract(r"(\d+)")[0]
        df["emp_length"] = pd.to_numeric(extracted, errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)

    impute_cols = ["dti", "emp_length", "risk_score"]
    imputation_values = {}
    for c in impute_cols:
        if c in df.columns and df[c].isna().sum() > 0:
            median_val = df[c].median()
            imputation_values[c] = float(median_val)
            df[c] = df[c].fillna(median_val)
    log["imputation_values"] = imputation_values

    for c in ("loan_amnt_requested", "dti"):
        if c in df.columns:
            lo, hi = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(lower=lo, upper=hi)

    logger.info("Stage A cleaning complete: %s -> %s rows, %d columns",
                before, len(df), df.shape[1])
    return df, log
