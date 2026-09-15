"""
run_pipeline.py
================
End-to-end orchestration of both stages described in Chapter Three.

Stage B (default prediction, accepted loans) runs unconditionally: the
fallback data source (data_loader.load_accepted_data) guarantees it always
has real data to work with.

Stage A (acceptance/rejection) requires the real rejected-application file
from Kaggle. If it is not available, this script logs a clear message,
skips Stage A, and still completes Stage B -- it never fabricates Stage A
results.

Usage
-----
    python run_pipeline.py
"""

import json
import logging

from sklearn.model_selection import train_test_split

from src import config, data_loader, preprocessing, feature_engineering, models
from src.evaluation import plot_confusion_matrix, plot_roc_curves
from src.explainability import run_shap_analysis, run_lime_analysis
from fintech_prescriptive_recourse import evaluate_fpr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_stage_b() -> dict:
    logger.info("=" * 70)
    logger.info("STAGE B: Default Prediction (accepted loans)")
    logger.info("=" * 70)

    raw = data_loader.load_accepted_data()
    clean_df, clean_log = preprocessing.clean_accepted_data(raw)
    fe_df = feature_engineering.engineer_stage_b_features(clean_df)

    numeric_features, categorical_features = feature_engineering.get_stage_b_feature_lists(fe_df)
    X = fe_df[numeric_features + categorical_features]
    y = fe_df[feature_engineering.STAGE_B_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    preprocessor = feature_engineering.build_stage_b_preprocessor(numeric_features, categorical_features)
    fitted, metrics = models.train_and_evaluate(
        preprocessor, X_train, y_train, X_test, y_test, stage_name="stage_b"
    )

    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    best_pipeline = fitted[best_name]

    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name, "stage_b")
    plot_roc_curves(fitted, metrics, X_test, y_test, "stage_b")

    explainer, feature_names, shap_importance = run_shap_analysis(
        best_pipeline, X_test, categorical_features, "stage_b"
    )
    run_lime_analysis(
        best_pipeline, X_train, X_test, categorical_features,
        "stage_b", class_names=["Fully Paid", "Default"],
    )

    def recompute_stage_b(row):
        row = row.copy()
        row["loan_to_income"] = row["loan_amnt"] / (row["annual_inc"] + 1)
        row["installment_burden"] = row["loan_amnt"] / max(row["term_months"], 1)
        row["revol_delinq_interaction"] = row["revol_util"] * (row["delinq_2yrs"] + 1)
        return row

    proba_all = best_pipeline.predict_proba(X_test)[:, 1]
    declined_mask = proba_all >= config.APPROVAL_THRESHOLD
    declined_sample = X_test[declined_mask].sample(
        n=min(100, declined_mask.sum()), random_state=config.SEED
    )

    fpr_summary = evaluate_fpr(
        declined_sample, best_pipeline, explainer,
        best_pipeline.named_steps["prep"], feature_names,
        numeric_features + categorical_features,
        config.FPR_ACTIONABLE_FEATURES_STAGE_B, recompute_stage_b, "stage_b",
    )

    result = {
        "cleaning_log": clean_log,
        "model_metrics": metrics,
        "best_model": best_name,
        "shap_top_10": shap_importance.head(10).round(4).to_dict(),
        "fpr_evaluation": fpr_summary,
    }
    with open(config.REPORTS_DIR / "stage_b_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def run_stage_a() -> dict | None:
    logger.info("=" * 70)
    logger.info("STAGE A: Acceptance Prediction (accepted + rejected)")
    logger.info("=" * 70)

    try:
        rejected_raw = data_loader.load_rejected_data()
    except FileNotFoundError as e:
        logger.warning("Skipping Stage A: %s", e)
        return None

    accepted_raw = data_loader.load_accepted_data()
    accepted_clean, _ = preprocessing.clean_accepted_data(accepted_raw)
    rejected_clean, clean_log = preprocessing.clean_rejected_data(rejected_raw)

    combined = feature_engineering.engineer_stage_a_features(accepted_clean, rejected_clean)
    X = combined[feature_engineering.STAGE_A_NUMERIC + feature_engineering.STAGE_A_CATEGORICAL]
    y = combined[feature_engineering.STAGE_A_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    preprocessor = feature_engineering.build_stage_a_preprocessor()
    fitted, metrics = models.train_and_evaluate(
        preprocessor, X_train, y_train, X_test, y_test, stage_name="stage_a"
    )
    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    best_pipeline = fitted[best_name]

    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name, "stage_a")
    plot_roc_curves(fitted, metrics, X_test, y_test, "stage_a")

    explainer, feature_names, shap_importance = run_shap_analysis(
        best_pipeline, X_test, feature_engineering.STAGE_A_CATEGORICAL, "stage_a"
    )
    run_lime_analysis(
        best_pipeline, X_train, X_test, feature_engineering.STAGE_A_CATEGORICAL,
        "stage_a", class_names=["Rejected", "Accepted"],
    )

    def recompute_stage_a(row):
        return row  # no dependent engineered features on the Stage A side

    proba_all = best_pipeline.predict_proba(X_test)[:, 1]
    # NOTE: Stage A's target is P(accepted); "affected" applicants for
    # recourse purposes are those predicted rejected, i.e. low proba.
    rejected_mask = proba_all < (1 - config.APPROVAL_THRESHOLD)
    rejected_sample = X_test[rejected_mask].sample(
        n=min(100, rejected_mask.sum()), random_state=config.SEED
    )

    fpr_summary = evaluate_fpr(
        rejected_sample, best_pipeline, explainer,
        best_pipeline.named_steps["prep"], feature_names,
        feature_engineering.STAGE_A_NUMERIC + feature_engineering.STAGE_A_CATEGORICAL,
        config.FPR_ACTIONABLE_FEATURES_STAGE_A, recompute_stage_a, "stage_a",
    )

    result = {
        "cleaning_log": clean_log,
        "model_metrics": metrics,
        "best_model": best_name,
        "shap_top_10": shap_importance.head(10).round(4).to_dict(),
        "fpr_evaluation": fpr_summary,
    }
    with open(config.REPORTS_DIR / "stage_a_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    stage_b_result = run_stage_b()
    stage_a_result = run_stage_a()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Stage B best model: {stage_b_result['best_model']}")
    print(f"Stage B ROC-AUC: {stage_b_result['model_metrics'][stage_b_result['best_model']]['roc_auc']}")
    if stage_a_result:
        print(f"Stage A best model: {stage_a_result['best_model']}")
        print(f"Stage A ROC-AUC: {stage_a_result['model_metrics'][stage_a_result['best_model']]['roc_auc']}")
    else:
        print("Stage A: SKIPPED (rejected-application data not available -- see log above)")