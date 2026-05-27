# XGBoost Fraud Detection — DBS Bank

A production-style fraud detection system built during my time as an Associate Data Scientist at DBS Bank. The core challenge was a heavily imbalanced dataset — roughly 1 in 200 transactions was fraudulent — and the need to explain model decisions to a non-technical compliance team.

---

## What This Does

Raw transaction logs come in. A PySpark feature pipeline computes velocity, behavioral, and temporal features. An XGBoost classifier scores each transaction. SHAP values explain every prediction. The model is tracked in MLflow and deployed via SageMaker.

---

## Architecture

```
Raw Transaction Logs (S3)
        │
        ▼
┌───────────────────────────┐
│   PySpark Feature Pipeline │
│                            │
│   Velocity features        │
│   Rolling time averages    │
│   Merchant category flags  │
│   Device consistency       │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Data Preparation         │
│                            │
│   SMOTE oversampling       │
│   Train/validation split   │
│   Stratified by class      │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   XGBoost Training         │
│                            │
│   Precision-recall tuning  │
│   Cross-validation         │
│   Threshold calibration    │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Evaluation + Explainability│
│                             │
│   SHAP TreeExplainer        │
│   AUC-ROC, F1, Precision    │
│   Confusion matrix          │
└─────────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   MLflow + SageMaker        │
│                             │
│   Experiment tracking       │
│   Model registry            │
│   Endpoint deployment       │
└─────────────────────────────┘
```

---

## Key Design Decisions

### Why not remove outliers?
In fraud detection, extreme transaction values are often the fraud signal. Capping outliers with winsorization and creating binary outlier flags preserved the signal while stabilising the model.

### Why SMOTE over class weighting?
Both work. SMOTE was chosen because it generates synthetic minority samples rather than just reweighting. For this dataset and problem type, SMOTE gave more stable precision-recall curves across cross-validation folds.

### Why tune on precision-recall rather than accuracy?
With 1:200 class imbalance, a model that predicts "not fraud" for everything achieves 99.5% accuracy. Accuracy is meaningless here. The risk team cared about catching fraud (recall) without generating too many false positives (precision). The threshold was tuned to their specific operational tolerance.

### Why SHAP for explainability?
The compliance team needed to justify model decisions to regulators. "The model said so" is not acceptable. SHAP TreeExplainer gives per-prediction feature contributions that can be explained in plain terms — "this transaction was flagged because the amount was 8x higher than this customer's 30-day average."

---

## Repo Structure

```
fraud-detection-xgboost/
├── src/
│   ├── features/
│   │   ├── velocity.py         # Transaction velocity features
│   │   ├── rolling.py          # Rolling time-window aggregations
│   │   ├── merchant.py         # Merchant category features
│   │   └── pipeline.py         # Full PySpark feature pipeline
│   ├── models/
│   │   ├── train.py            # XGBoost training
│   │   ├── smote.py            # SMOTE oversampling
│   │   └── threshold.py        # Precision-recall threshold tuning
│   ├── evaluation/
│   │   ├── metrics.py          # AUC, F1, precision, recall
│   │   └── shap_explainer.py   # SHAP TreeExplainer
│   └── serving/
│       └── mlflow_registry.py  # MLflow + SageMaker registration
├── notebooks/
│   └── exploration.ipynb       # EDA and feature exploration
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Stack

| Layer | Tools |
|---|---|
| Feature engineering | PySpark, pandas, numpy |
| Oversampling | imbalanced-learn (SMOTE) |
| Modeling | XGBoost |
| Explainability | SHAP TreeExplainer |
| Evaluation | scikit-learn (AUC, F1, precision-recall) |
| Experiment tracking | MLflow |
| Model serving | Amazon SageMaker |

---

## Setup

```bash
git clone https://github.com/saichandrapodili/fraud-detection-xgboost
cd fraud-detection-xgboost
pip install -r requirements.txt
cp .env.example .env
# Add your AWS credentials in .env
python src/models/train.py
```

---

## About

Built during my first industry role as Associate Data Scientist at DBS Bank in Hyderabad. My responsibility was model development and feature engineering — the senior engineers handled production deployment and infrastructure monitoring. This repo is a cleaned-up version of that work for portfolio purposes.

MS in Data Science — University of Alabama at Birmingham  
[LinkedIn](https://linkedin.com/in/saichandrapodili) · [Email](mailto:saichandrakirantejap@gmail.com)
