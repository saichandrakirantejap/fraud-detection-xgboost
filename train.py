"""
train.py

XGBoost fraud detection model training.

The threshold tuning step is what makes the model actually useful.
Accuracy is meaningless on imbalanced data. The risk team gave us
a specific operational tolerance — they could handle reviewing N
flagged transactions per day. We tuned the threshold to hit their
precision requirement while maximising recall within that constraint.
"""

import mlflow
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    classification_report
)
from imblearn.over_sampling import SMOTE


def train_fraud_model(X_train, y_train, X_val, y_val, config: dict):
    """
    Train XGBoost fraud classifier with MLflow tracking.
    """
    with mlflow.start_run():
        mlflow.log_params(config)

        # SMOTE on training data only — never on validation
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            min_child_weight=config.get("min_child_weight", 5),
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42
        )

        model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        # Get probability scores
        y_proba = model.predict_proba(X_val)[:, 1]

        # AUC on validation
        auc = roc_auc_score(y_val, y_proba)
        mlflow.log_metric("val_auc", auc)

        # Find optimal threshold for the risk team's tolerance
        threshold = tune_threshold(y_val, y_proba, target_precision=0.80)
        mlflow.log_metric("optimal_threshold", threshold)

        # Log metrics at the chosen threshold
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        mlflow.log_metric("val_f1", f1)

        mlflow.xgboost.log_model(model, "model")

        print(f"AUC: {auc:.4f}")
        print(f"Threshold: {threshold:.3f}")
        print(classification_report(y_val, y_pred, target_names=["legit", "fraud"]))

        return model, threshold


def tune_threshold(y_true, y_proba, target_precision: float = 0.80) -> float:
    """
    Find the lowest threshold that achieves at least target_precision.

    We tune on precision-recall rather than accuracy because:
    - Accuracy is misleading with 1:200 imbalance
    - The risk team has a specific false positive tolerance
    - We want maximum recall subject to meeting precision requirement
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find thresholds where precision meets the target
    valid_thresholds = [
        (threshold, recall)
        for precision, recall, threshold in zip(precisions, recalls, thresholds)
        if precision >= target_precision
    ]

    if not valid_thresholds:
        # Fall back to 0.5 if target precision can't be met
        return 0.5

    # Among valid thresholds, pick the one with highest recall
    best_threshold = max(valid_thresholds, key=lambda x: x[1])[0]
    return float(best_threshold)


def cross_validate(X, y, config: dict, n_splits: int = 5) -> dict:
    """
    Stratified k-fold cross-validation.
    Stratified ensures each fold has the same fraud rate as the full dataset.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_fold_train, y_fold_train)

        model = XGBClassifier(**config, random_state=42)
        model.fit(X_resampled, y_resampled)

        y_proba = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_proba)
        auc_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.4f}")

    return {
        "mean_auc": np.mean(auc_scores),
        "std_auc": np.std(auc_scores)
    }
