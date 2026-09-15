"""
quick_data_check.py
====================
A fast, no-modelling sanity check: confirms both datasets load, reports
their shape and target balance, and flags anything that would break the
main pipeline before you spend time running it.
"""

from src import config, data_loader
from src.preprocessing import GOOD_LOAN_STATUSES, DEFAULT_LOAN_STATUSES


def check_accepted():
    df = data_loader.load_accepted_data()
    
    print(f"\n[Raw Ingestion] Accepted data: {df.shape}")
    
    # 1. Track data loss from dropping transitory states
    allowed_statuses = GOOD_LOAN_STATUSES.union(DEFAULT_LOAN_STATUSES)
    filtered_df = df[df["loan_status"].isin(allowed_statuses)].copy()
    
    # 2. Formally map to the 'is_default' binary target variable
    filtered_df["is_default"] = filtered_df["loan_status"].isin(DEFAULT_LOAN_STATUSES).astype(int)
    
    print(f"[Post-Filtering] Accepted data (Transitory removed): {filtered_df.shape}")
    print("\nData type counts:")
    print(filtered_df.dtypes.value_counts())
    
    if "is_default" in filtered_df.columns:
        print("\nTarget balance (is_default):")
        counts = filtered_df["is_default"].value_counts()
        pcts = filtered_df["is_default"].value_counts(normalize=True).round(4)
        balance_report = f"0 (Performing): {counts[0]} ({pcts[0] * 100:.2f}%)\n1 (Defaulted):  {counts[1]} ({pcts[1] * 100:.2f}%)"
        print(balance_report)
        
    print(f"\nMissing values (top 10):\n{filtered_df.isna().sum().sort_values(ascending=False).head(10)}")


def check_rejected():
    try:
        df = data_loader.load_rejected_data()
    except FileNotFoundError as e:
        print(f"\n[Stage A unavailable] {e}")
        return
    print(f"\nRejected data: {df.shape}")
    print("\nRejected columns:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK DATA CHECK (TARGET: IS_DEFAULT)")
    print("=" * 60)
    check_accepted()
    check_rejected()
