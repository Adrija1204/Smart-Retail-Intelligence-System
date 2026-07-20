"""SHAP exports for tree-based retail propensity models."""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def _positive_class_values(values: object) -> np.ndarray:
    """Normalize binary SHAP output to a two-dimensional positive-class array."""
    if isinstance(values, list):
        return np.asarray(values[1])
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, 1]
    return array


def _explanation_tables(
    values: np.ndarray, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert SHAP arrays into dashboard-ready global and local tables."""
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    contributions = (
        pd.DataFrame(values, columns=features.columns)
        .rename_axis("row_index")
        .reset_index()
        .melt(id_vars="row_index", var_name="feature", value_name="shap_value")
    )
    return importance.reset_index(drop=True), contributions


def explain_random_forest(
    model: RandomForestClassifier, features: pd.DataFrame, sample_size: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return global SHAP importance and local contributions for sampled rows."""
    sampled = features.head(sample_size).copy()
    explainer = shap.TreeExplainer(model)
    values = _positive_class_values(explainer.shap_values(sampled))
    return _explanation_tables(values, sampled)


def explain_logistic_pipeline(
    model: Pipeline, features: pd.DataFrame, sample_size: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return SHAP tables for a fitted scaled logistic-regression pipeline."""
    sampled = features.head(sample_size).copy()
    scaled_features = model.named_steps["scaler"].transform(sampled)
    classifier = model.named_steps["classifier"]
    explainer = shap.LinearExplainer(classifier, scaled_features)
    values = _positive_class_values(explainer.shap_values(scaled_features))
    return _explanation_tables(values, sampled)
