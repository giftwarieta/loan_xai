"""
models.py
=========
Trains and evaluates Logistic Regression, Random Forest, and XGBoost for
either stage. Both stages share this module; only the preprocessor and
class-imbalance direction differ (Chapter Three, Section 3.6).
"""

import logging

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)


def build_models(pos_weight: float) -> dict:
    """pos_weight = (# majority class) / (# minority class) in the training
    split, used for XGBoost's scale_pos_weight and to set class_weight on
    the other two models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.SEED
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", **config.MODEL_PARAMS["random_forest"]
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=pos_weight, **config.MODEL_PARAMS["xgboost"]
        ),
    }


def train_and_evaluate(preprocessor, X_train, y_train, X_test, y_test,
                        stage_name: str) -> tuple[dict, dict]:
    """Fits all three models inside `preprocessor` pipelines, evaluates each
    on the test set, and returns (fitted_pipelines, metrics_dict)."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = build_models(pos_weight)

    fitted, metrics = {}, {}
    for name, clf in models.items():
        # Each pipeline needs an independent fitted transformer.  Reusing one
        # instance mutates previously fitted pipelines when the next model
        # trains and can make saved artifacts inconsistent.
        pipe = Pipeline([("prep", clone(preprocessor)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = compute_metrics(y_test, y_pred, y_proba)
        logger.info("[%s] %s: %s", stage_name, name, metrics[name])

    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    out_path = config.MODELS_DIR / f"{stage_name}_best_model.joblib"
    joblib.dump(fitted[best_name], out_path)
    joblib.dump(fitted, config.MODELS_DIR / f"{stage_name}_all_models.joblib")
    logger.info("[%s] Best model: %s (saved to %s)", stage_name, best_name, out_path)

    return fitted, metrics
