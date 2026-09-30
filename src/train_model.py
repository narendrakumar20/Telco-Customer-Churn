"""
train_model.py
--------------
Trains a Random Forest Classifier on the Telco churn dataset
and optionally registers it in the Snowflake Model Registry.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import FEATURE_COLS, TARGET_COL, DATABASE, SCHEMA


# ──────────────────────────────────────────────────────────
# Model Config
# ──────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth"   : 10,
    "random_state": 42,
    "n_jobs"      : -1,
}


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, params: dict = None):
    """
    Train a Random Forest Classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Target labels for training.
    params : dict, optional
        Hyperparameters to override RF_PARAMS defaults.

    Returns
    -------
    model : sklearn RandomForestClassifier (fitted)
    """
    cfg = {**RF_PARAMS, **(params or {})}
    model = RandomForestClassifier(**cfg)
    model.fit(X_train, y_train)
    print("✅ Random Forest model trained successfully!")
    print(f"   n_estimators={cfg['n_estimators']} | max_depth={cfg['max_depth']}")
    return model


# ──────────────────────────────────────────────────────────
# Feature Importance
# ──────────────────────────────────────────────────────────
def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract and sort feature importances from the trained model.

    Returns
    -------
    pd.DataFrame with columns [FEATURE, IMPORTANCE]
    """
    fi = pd.DataFrame({
        "FEATURE"   : FEATURE_COLS,
        "IMPORTANCE": model.feature_importances_,
    }).sort_values("IMPORTANCE", ascending=False).reset_index(drop=True)

    print("\n📊 Feature Importances:")
    print(fi.to_string(index=False))
    return fi


# ──────────────────────────────────────────────────────────
# Snowflake Model Registry
# ──────────────────────────────────────────────────────────
def register_model(session, model, X_train, metrics: dict, version: str = "V1"):
    """
    Register the trained sklearn model in the Snowflake Model Registry.

    Parameters
    ----------
    session  : Snowflake Snowpark session
    model    : Fitted sklearn model
    X_train  : pd.DataFrame — sample input for schema inference
    metrics  : dict — evaluation metrics to attach (accuracy, precision, etc.)
    version  : str  — model version tag (default "V1")

    Returns
    -------
    model_ref : Snowflake model registry reference
    """
    from snowflake.ml.registry import Registry

    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )

    model_ref = registry.log_model(
        model,
        model_name="CUSTOMER_CHURN_RF",
        version_name=version,
        sample_input_data=X_train.head(10),
        metrics={k: float(v) for k, v in metrics.items()},
        conda_dependencies=["scikit-learn"],
        comment=f"Telco Customer Churn — Random Forest ({version})",
    )

    print(f"✅ Model registered → CUSTOMER_CHURN_RF / {version}")
    return model_ref


def register_model_warehouse(session, model, X_train, metrics: dict, version: str = "V2"):
    """
    Register model with WAREHOUSE_ONLY inference target for SQL-based scoring.
    """
    from snowflake.ml.registry import Registry
    from snowflake.ml.model import target_platform

    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )

    model_ref = registry.log_model(
        model,
        model_name="CUSTOMER_CHURN_RF",
        version_name=version,
        sample_input_data=X_train.head(10),
        metrics={k: float(v) for k, v in metrics.items()},
        conda_dependencies=["scikit-learn"],
        target_platforms=target_platform.WAREHOUSE_ONLY,
        comment=f"Telco Customer Churn — Warehouse inference ({version})",
    )

    print(f"✅ Model V2 registered (Warehouse Only) → CUSTOMER_CHURN_RF / {version}")
    return model_ref
