"""
evaluation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from . import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, stage_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name} ({stage_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out = config.FIGURES_DIR / f"confusion_matrix_{stage_name}_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_roc_curves(fitted_pipelines: dict, metrics: dict, X_test, y_test, stage_name: str):
    plt.figure(figsize=(7, 6))
    for name, pipe in fitted_pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {stage_name}")
    plt.legend()
    plt.tight_layout()
    out = config.FIGURES_DIR / f"roc_curves_{stage_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out
