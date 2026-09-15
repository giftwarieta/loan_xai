"""
explainability.py
==================
SHAP (global + instance-level) and LIME (local) explanations, applied
identically to whichever stage's best model is passed in. Chapter Three,
Section 3.8.
"""

import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.neural_network import MLPClassifier
from lime.lime_tabular import LimeTabularExplainer

from . import config

logger = logging.getLogger(__name__)


def _get_feature_names(preprocessor, categorical_cols) -> list:
    numeric_cols = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    return list(numeric_cols) + cat_names


def _positive_class_values(values) -> np.ndarray:
    """Normalise SHAP's version/model-specific binary output to (n, p)."""
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        # SHAP 0.45+ commonly returns (samples, features, classes).
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    return values


def _build_shap_explainer(clf, background):
    """Choose an exact or model-agnostic explainer compatible with the fitted classifier."""
    if hasattr(clf, "estimators_") or hasattr(clf, "get_booster"):
        return shap.TreeExplainer(clf)
    if hasattr(clf, "coef_"):
        return shap.LinearExplainer(clf, background)
    if isinstance(clf, MLPClassifier) or hasattr(clf, "coefs_"):
        # Use shap.KernelExplainer for Neural Networks / MLPs with a summarized background for performance
        background_summary = shap.kmeans(background, min(50, len(background))) if len(background) > 50 else background
        return shap.KernelExplainer(clf.predict_proba, background_summary)
    raise TypeError(f"Unsupported classifier for SHAP analysis: {type(clf).__name__}")


def run_shap_analysis(best_pipeline, X_test, categorical_cols, stage_name: str,
                       sample_size: int = 2000):
    preprocessor = best_pipeline.named_steps["prep"]
    clf = best_pipeline.named_steps["clf"]
    feature_names = _get_feature_names(preprocessor, categorical_cols)

    X_transformed = preprocessor.transform(X_test)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    sample = X_df.sample(n=min(sample_size, len(X_df)), random_state=config.SEED)
    explainer = _build_shap_explainer(clf, sample)
    shap_values = _positive_class_values(explainer.shap_values(sample))

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_summary_bar_{stage_name}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"shap_summary_beeswarm_{stage_name}.png", dpi=150)
    plt.close()

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(config.REPORTS_DIR / f"shap_feature_importance_{stage_name}.csv")

    return explainer, feature_names, importance


def run_lime_analysis(best_pipeline, X_train, X_test, categorical_cols,
                       stage_name: str, n_samples: int = 3, class_names=None):
    preprocessor = best_pipeline.named_steps["prep"]
    clf = best_pipeline.named_steps["clf"]
    feature_names = _get_feature_names(preprocessor, categorical_cols)

    train_transformed = preprocessor.transform(X_train)
    if hasattr(train_transformed, "toarray"):
        train_transformed = train_transformed.toarray()

    test_transformed = preprocessor.transform(X_test)
    if hasattr(test_transformed, "toarray"):
        test_transformed = test_transformed.toarray()
    X_test_df = pd.DataFrame(test_transformed, columns=feature_names)

    explainer = LimeTabularExplainer(
        training_data=train_transformed, feature_names=feature_names,
        class_names=class_names or ["Negative", "Positive"],
        mode="classification", random_state=config.SEED,
    )

    n_samples = min(n_samples, len(X_test_df))
    if n_samples == 0:
        logger.warning("[%s] Skipping LIME: test set is empty.", stage_name)
        return {}
    sample_idx = X_test_df.sample(n=n_samples, random_state=config.SEED).index
    summaries = {}
    for i, idx in enumerate(sample_idx):
        exp = explainer.explain_instance(X_test_df.loc[idx].values, clf.predict_proba, num_features=8)
        summaries[f"instance_{i+1}"] = exp.as_list()
        exp.save_to_file(str(config.REPORTS_DIR / f"lime_{stage_name}_instance_{i+1}.html"))

    with open(config.REPORTS_DIR / f"lime_summaries_{stage_name}.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    return summaries