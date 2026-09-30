"""
evaluate_model.py
-----------------
Computes classification metrics and confusion matrix
for the trained Telco Customer Churn model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ──────────────────────────────────────────────────────────
# Core Evaluation
# ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Generate predictions and compute key classification metrics.

    Parameters
    ----------
    model  : Fitted sklearn classifier
    X_test : pd.DataFrame — test feature matrix
    y_test : pd.Series   — true test labels

    Returns
    -------
    metrics : dict with keys [accuracy, precision, recall, f1_score]
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1_score" : f1_score(y_test, y_pred),
    }

    print("\n📊 Model Evaluation Results")
    print("─" * 35)
    for name, val in metrics.items():
        print(f"   {name.capitalize():<12}: {val:.4f}")

    return metrics, y_pred


# ──────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────
def print_confusion_matrix(y_test, y_pred):
    """
    Print confusion matrix with labeled interpretation.

    Parameters
    ----------
    y_test : true labels
    y_pred : predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Confusion Matrix")
    print("─" * 35)
    print(cm)
    print("\n  Interpretation:")
    print(f"  TN (True  Negative) : {cm[0][0]}")
    print(f"  FP (False Positive) : {cm[0][1]}")
    print(f"  FN (False Negative) : {cm[1][0]}")
    print(f"  TP (True  Positive) : {cm[1][1]}")

    return cm


# ──────────────────────────────────────────────────────────
# Full Report
# ──────────────────────────────────────────────────────────
def full_classification_report(y_test, y_pred):
    """Print sklearn's classification report (precision/recall/F1 per class)."""
    print("\n📋 Classification Report")
    print("─" * 50)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
