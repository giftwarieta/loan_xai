"""
fintech_prescriptive_recourse.py
==================================
FintechPrescriptiveRecourse (FPR) -- this thesis's original contribution.

Implements exactly the algorithm formalised in Chapter Three, Section 3.9:
a SHAP-guided, cost-weighted, actionability-constrained greedy coordinate
search over a bounded mixed-integer feasibility problem (Section 3.9.1),
approximated by Algorithm 1 (Section 3.9.2).

This module is stage-agnostic: pass it a fitted pipeline, a SHAP explainer,
an actionable-feature config (from config.py), and a "recompute engineered
features" function, and it works identically for Stage A (acceptance) or
Stage B (default).
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _positive_class_shap_values(values) -> np.ndarray:
    """Return one SHAP vector for the positive class across SHAP versions."""
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    return values


def _predict_proba_row(row: pd.Series, pipeline, feature_order: list) -> float:
    row_df = pd.DataFrame([row])[feature_order]
    return pipeline.predict_proba(row_df)[0, 1]


def fintech_prescriptive_recourse(
    applicant_row: pd.Series,
    pipeline,
    shap_explainer,
    preprocessor,
    feature_names: list,
    feature_order: list,
    actionable_features: dict,
    recompute_engineered: Callable[[pd.Series], pd.Series],
    threshold: float = config.APPROVAL_THRESHOLD,
    max_iterations: int = config.FPR_MAX_ITERATIONS,
    weighted: bool = True,
) -> dict:
    """
    Implements Algorithm 1 (Chapter Three, Section 3.9.2).

    Parameters
    ----------
    applicant_row : the applicant's raw (pre-preprocessing) feature values
    pipeline       : fitted sklearn Pipeline (preprocessor + classifier)
    shap_explainer : fitted shap.TreeExplainer for this pipeline's classifier
    preprocessor   : the pipeline's fitted ColumnTransformer (for SHAP input)
    feature_names  : post-transformation feature names (for SHAP indexing)
    feature_order  : raw column order the pipeline expects at predict time
    actionable_features : dict of {feature: (direction, step, cost, max_steps)}
                           -- see config.FPR_ACTIONABLE_FEATURES_STAGE_{A,B}
    recompute_engineered : function that recomputes dependent engineered
                           features after a raw feature changes (g(x') in
                           Section 3.9.1)
    weighted       : if False, uses a fixed (non-SHAP) priority order --
                     the ablation baseline reported in Chapter Four.

    Returns
    -------
    dict with success flag, original/final probability, features changed,
    total cost, and iterations used (mirrors the notation in Section 3.9).
    """
    row = applicant_row.copy()
    original_proba = _predict_proba_row(row, pipeline, feature_order)

    if original_proba < threshold:
        return {"already_favourable": True}

    row_transformed = preprocessor.transform(pd.DataFrame([row])[feature_order])
    if hasattr(row_transformed, "toarray"):
        row_transformed = row_transformed.toarray()
    instance_shap = _positive_class_shap_values(
        shap_explainer.shap_values(row_transformed)
    )[0]
    shap_series = pd.Series(instance_shap, index=feature_names)

    if weighted:
        priority = sorted(
            actionable_features.keys(),
            key=lambda f: abs(shap_series.get(f, 0.0)),
            reverse=True,
        )
    else:
        priority = list(actionable_features.keys())

    steps_used = {f: 0 for f in actionable_features}
    total_cost = 0.0
    changes: dict = {}
    proba = original_proba
    t = 0

    for t in range(1, max_iterations + 1):
        if proba < threshold:
            break
        improved = False
        for feat in priority:
            direction, step, cost_weight, max_steps = actionable_features[feat]
            if steps_used[feat] >= max_steps:
                continue
            trial = row.copy()
            new_val = trial[feat] + direction * step
            trial[feat] = max(new_val, 0)
            trial = recompute_engineered(trial)
            trial_proba = _predict_proba_row(trial, pipeline, feature_order)

            if trial_proba < proba:
                row = trial
                proba = trial_proba
                total_cost += cost_weight
                steps_used[feat] += 1
                changes[feat] = changes.get(feat, 0) + direction * step
                improved = True
                if proba < threshold:
                    break
        if not improved:
            break

    success = proba < threshold
    return {
        "already_favourable": False,
        "success": bool(success),
        "original_proba": round(float(original_proba), 4),
        "final_proba": round(float(proba), 4),
        "n_features_changed": len(changes),
        "changes": {k: round(float(v), 2) for k, v in changes.items()},
        "total_cost": round(float(total_cost), 2),
        "iterations_used": t,
    }


def evaluate_fpr(
    declined_sample: pd.DataFrame,
    pipeline, shap_explainer, preprocessor, feature_names, feature_order,
    actionable_features, recompute_engineered, stage_name: str,
) -> dict:
    """Runs FPR (weighted) and the unweighted ablation baseline over a
    sample of affected applicants, returning the Chapter Four comparison
    table (success rate, avg. features changed, avg. cost, avg. proba
    reduction) for both conditions."""
    fpr_results, baseline_results = [], []
    for _, row in declined_sample.iterrows():
        fpr_results.append(fintech_prescriptive_recourse(
            row, pipeline, shap_explainer, preprocessor, feature_names,
            feature_order, actionable_features, recompute_engineered, weighted=True))
        baseline_results.append(fintech_prescriptive_recourse(
            row, pipeline, shap_explainer, preprocessor, feature_names,
            feature_order, actionable_features, recompute_engineered, weighted=False))

    def summarise(results):
        ok = [r for r in results if r.get("success")]
        n = len(results)
        return {
            "n_evaluated": n,
            "success_rate": round(len(ok) / n, 4) if n else None,
            "avg_features_changed": round(np.mean([r["n_features_changed"] for r in ok]), 2) if ok else None,
            "avg_cost": round(np.mean([r["total_cost"] for r in ok]), 2) if ok else None,
            "avg_proba_reduction": round(np.mean([r["original_proba"] - r["final_proba"] for r in ok]), 4) if ok else None,
        }

    summary = {
        "stage": stage_name,
        "fpr_summary": summarise(fpr_results),
        "baseline_summary": summarise(baseline_results),
    }
    logger.info("[%s] FPR evaluation: %s", stage_name, summary)
    return summary
