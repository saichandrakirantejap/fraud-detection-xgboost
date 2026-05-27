"""
shap_explainer.py

SHAP TreeExplainer for fraud model explainability.

The compliance team needed to justify model decisions to regulators.
SHAP gives per-prediction feature contributions that translate into
plain language explanations — "flagged because amount was 8x higher
than 30-day average, and merchant category was null."

Three outputs:
1. Global feature importance — which features matter most overall
2. Local explanation — why this specific transaction was flagged
3. Dependence plots — how a feature's impact changes with its value
"""

import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt


def build_explainer(model, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """Build SHAP TreeExplainer. Fit on training data."""
    return shap.TreeExplainer(model)


def explain_prediction(
    explainer: shap.TreeExplainer,
    transaction: pd.DataFrame,
    feature_names: list
) -> dict:
    """
    Explain a single transaction prediction.

    Returns a dict mapping feature names to their SHAP contribution,
    sorted by absolute impact. Positive values push toward fraud,
    negative toward legitimate.
    """
    shap_values = explainer.shap_values(transaction)

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        fraud_shap = shap_values[1][0]
    else:
        fraud_shap = shap_values[0]

    contributions = dict(zip(feature_names, fraud_shap))

    # Sort by absolute impact
    return dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )


def plot_global_importance(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    output_path: str = "shap_importance.png"
) -> None:
    """
    Bar plot of mean absolute SHAP values — overall feature importance.
    Saved to file for inclusion in reports.
    """
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plain_language_explanation(contributions: dict, top_n: int = 3) -> str:
    """
    Convert SHAP contributions into a plain language explanation
    for the compliance team.
    """
    top_features = list(contributions.items())[:top_n]
    reasons = []

    for feature, value in top_features:
        direction = "increased" if value > 0 else "decreased"
        reasons.append(f"'{feature}' {direction} the fraud score")

    return "This transaction was flagged because: " + "; ".join(reasons) + "."
