"""
data_preprocessing.py
----------------------
Handles Snowflake data ingestion, feature selection,
train/test splitting, and conversion to pandas for
the Telco Customer Churn ML project.
"""

from snowflake.snowpark.context import get_active_session

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
DATABASE = "ML_PROJECT"
SCHEMA   = "CHURN"
TABLE    = "CHURN_DATASET"

FEATURE_COLS = [
    "SENIORCITIZEN",
    "TENURE",
    "MONTHLYCHARGES",
    "TOTALCHARGES",
    "GENDER_MALE",
    "PARTNER_YES",
    "DEPENDENTS_YES",
    "PHONE_SERVICE",
    "PAPERLESS_BILLING",
    "CONTRACT_MONTH_TO_MONTH",
    "CONTRACT_ONE_YEAR",
    "CONTRACT_TWO_YEAR",
    "INTERNET_DSL",
    "INTERNET_FIBER",
    "INTERNET_NONE",
]

TARGET_COL = "CHURN_TARGET"


# ──────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────
def get_session():
    """Return the active Snowflake Snowpark session."""
    session = get_active_session()
    print("✅ Snowflake connection successful!")
    return session


# ──────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────
def load_dataset(session):
    """
    Load the prepared churn dataset from Snowflake.

    Parameters
    ----------
    session : snowflake.snowpark.Session

    Returns
    -------
    df : snowflake.snowpark.DataFrame
    """
    df = session.table(f"{DATABASE}.{SCHEMA}.{TABLE}")
    print(f"✅ Dataset loaded → {df.count()} rows")
    print(f"   Columns: {df.columns}")
    return df


# ──────────────────────────────────────────────────────────
# EDA Helper
# ──────────────────────────────────────────────────────────
def check_class_distribution(train_df):
    """Display churn class distribution in the training split."""
    print("\n📊 Churn Class Distribution (Train):")
    train_df.group_by(TARGET_COL).count().sort(TARGET_COL).show()


# ──────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────
def split_data(df):
    """
    Separate the Snowpark DataFrame into train and test splits
    using the DATA_SPLIT column (values: 'TRAIN' | 'TEST').

    Returns
    -------
    train_df, test_df : Snowpark DataFrames
    """
    train_df = df.filter(df["DATA_SPLIT"] == "TRAIN")
    test_df  = df.filter(df["DATA_SPLIT"] == "TEST")

    print(f"✅ Train rows : {train_df.count()}")
    print(f"   Test rows  : {test_df.count()}")
    return train_df, test_df


# ──────────────────────────────────────────────────────────
# Pandas Conversion
# ──────────────────────────────────────────────────────────
def to_pandas(train_df, test_df):
    """
    Convert Snowpark DataFrames to pandas arrays for sklearn training.

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas DataFrame / Series
    """
    train_data = train_df.select(FEATURE_COLS + [TARGET_COL]).to_pandas()
    test_data  = test_df.select(FEATURE_COLS + [TARGET_COL]).to_pandas()

    X_train = train_data[FEATURE_COLS]
    y_train = train_data[TARGET_COL]
    X_test  = test_data[FEATURE_COLS]
    y_test  = test_data[TARGET_COL]

    print(f"✅ X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
