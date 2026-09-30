"""
prediction.py
-------------
Saves model predictions and feature importances back to Snowflake,
and runs inference via the Snowflake Model Registry.
"""

import pandas as pd

from data_preprocessing import DATABASE, SCHEMA


# ──────────────────────────────────────────────────────────
# Save Predictions to Snowflake
# ──────────────────────────────────────────────────────────
def save_predictions(session, test_data: pd.DataFrame, y_pred):
    """
    Append predicted churn labels to the test dataframe
    and write the result to ML_PROJECT.CHURN.CHURN_PREDICTIONS.

    Parameters
    ----------
    session   : Snowflake Snowpark session
    test_data : pd.DataFrame — original test features + true labels
    y_pred    : array-like  — predicted churn labels
    """
    prediction_data = test_data.copy()
    prediction_data["PREDICTED_CHURN"] = y_pred

    session.write_pandas(
        prediction_data,
        table_name="CHURN_PREDICTIONS",
        database=DATABASE,
        schema=SCHEMA,
        auto_create_table=True,
        overwrite=True,
    )

    print(f"✅ Predictions saved → {DATABASE}.{SCHEMA}.CHURN_PREDICTIONS")
    print(f"   Rows saved: {len(prediction_data)}")


# ──────────────────────────────────────────────────────────
# Save Feature Importances to Snowflake
# ──────────────────────────────────────────────────────────
def save_feature_importance(session, feature_importance: pd.DataFrame):
    """
    Write the feature importance table to Snowflake.

    Parameters
    ----------
    session            : Snowflake Snowpark session
    feature_importance : pd.DataFrame with [FEATURE, IMPORTANCE] columns
    """
    session.write_pandas(
        feature_importance,
        table_name="FEATURE_IMPORTANCE",
        database=DATABASE,
        schema=SCHEMA,
        auto_create_table=True,
        overwrite=True,
    )

    print(f"✅ Feature importance saved → {DATABASE}.{SCHEMA}.FEATURE_IMPORTANCE")


# ──────────────────────────────────────────────────────────
# Registry Inference
# ──────────────────────────────────────────────────────────
def run_registry_inference(session, X_test: pd.DataFrame,
                            model_name: str = "CUSTOMER_CHURN_RF",
                            version: str = "V1",
                            n_samples: int = 10):
    """
    Load the registered model from Snowflake Model Registry
    and run predictions on a sample of test data.

    Parameters
    ----------
    session    : Snowflake Snowpark session
    X_test     : pd.DataFrame — test features
    model_name : str — registry model name
    version    : str — model version (e.g. "V1", "V2")
    n_samples  : int — number of rows to score

    Returns
    -------
    predictions : pd.DataFrame — registry model predictions
    """
    from snowflake.ml.registry import Registry

    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=SCHEMA,
    )

    registered_model = registry.get_model(model_name).version(version)
    predictions = registered_model.run(
        X_test.head(n_samples),
        function_name="predict",
    )

    print(f"✅ Registry inference successful ({model_name} / {version})")
    print(predictions)
    return predictions
