"""
deep_learning.py
================
Multi-Layer Perceptron (MLP) classifier, the fourth model named in the
thesis's objectives (Chapter One). Kept separate from models.py since it is
trained and tuned differently in practice (early stopping, learning-curve
monitoring) even though it plugs into the same Pipeline/metrics interface.
"""

import logging

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from . import config
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)


def train_mlp(preprocessor, X_train, y_train, X_test, y_test, stage_name: str):
    clf = MLPClassifier(**config.MODEL_PARAMS["mlp"])
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info("[%s] MLP: %s", stage_name, metrics)
    return pipe, metrics
