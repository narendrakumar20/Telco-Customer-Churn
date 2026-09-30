# 📞 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Snowflake](https://img.shields.io/badge/Snowflake-ML-29B5E8?logo=snowflake)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> Predict which telecom customers are likely to churn using a **Random Forest Classifier** trained on Snowflake data and registered in the **Snowflake Model Registry**.

---

## 📁 Project Structure

```
telco-customer-churn/
│
├── notebooks/
│   └── CUSTOMER_CHURN_ML.ipynb     # End-to-end ML workflow notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, splitting, pandas conversion
│   ├── train_model.py              # Model training & Snowflake registry
│   ├── evaluate_model.py           # Metrics, confusion matrix, report
│   └── prediction.py              # Save predictions & run registry inference
│
├── models/                         # Placeholder for local model artifacts
│
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

```
Snowflake (ML_PROJECT.CHURN)
        │
        ├── CHURN_DATASET          ← Raw + engineered features
        │
        ├── CHURN_PREDICTIONS      ← Model output (per customer)
        │
        ├── FEATURE_IMPORTANCE     ← RF feature importances
        │
        └── Model Registry
              └── CUSTOMER_CHURN_RF
                    ├── V1 (Notebook inference)
                    └── V2 (Warehouse-only inference)
```

---

## 🔍 Dataset Features

| Feature | Description |
|---|---|
| `SENIORCITIZEN` | Whether the customer is a senior citizen (0/1) |
| `TENURE` | Number of months the customer has stayed |
| `MONTHLYCHARGES` | Monthly subscription fee |
| `TOTALCHARGES` | Total charges over tenure |
| `GENDER_MALE` | Gender encoded (1 = Male) |
| `PARTNER_YES` | Has a partner (1 = Yes) |
| `DEPENDENTS_YES` | Has dependents (1 = Yes) |
| `PHONE_SERVICE` | Phone service enrolled (1 = Yes) |
| `PAPERLESS_BILLING` | Paperless billing (1 = Yes) |
| `CONTRACT_*` | Contract type (Month-to-Month / One Year / Two Year) |
| `INTERNET_*` | Internet service type (DSL / Fiber / None) |
| `CHURN_TARGET` | **Target** — 1 = Churned, 0 = Retained |

---

## 🚀 Workflow

### 1️⃣ Data Preprocessing
```python
from src.data_preprocessing import get_session, load_dataset, split_data, to_pandas

session  = get_session()
df       = load_dataset(session)
train_df, test_df = split_data(df)
X_train, X_test, y_train, y_test = to_pandas(train_df, test_df)
```

### 2️⃣ Model Training
```python
from src.train_model import train_random_forest, get_feature_importance

model = train_random_forest(X_train, y_train)
fi    = get_feature_importance(model)
```

### 3️⃣ Model Evaluation
```python
from src.evaluate_model import evaluate_model, print_confusion_matrix

metrics, y_pred = evaluate_model(model, X_test, y_test)
print_confusion_matrix(y_test, y_pred)
```

### 4️⃣ Save Predictions
```python
from src.prediction import save_predictions, save_feature_importance

save_predictions(session, test_data, y_pred)
save_feature_importance(session, fi)
```

### 5️⃣ Register Model
```python
from src.train_model import register_model, register_model_warehouse

register_model(session, model, X_train, metrics, version="V1")
register_model_warehouse(session, model, X_train, metrics, version="V2")
```

---

## 📊 Model Results

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| Precision | ~67% |
| Recall | ~53% |
| F1 Score | ~59% |

> Results vary based on data split and hyperparameter tuning.

---

## ⚙️ Setup & Installation

### Prerequisites
- Snowflake account with `ML_PROJECT.CHURN` database/schema set up
- Python 3.10+

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
Open `notebooks/CUSTOMER_CHURN_ML.ipynb` in **Snowflake Notebooks** or a local Jupyter environment connected to Snowflake.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Snowflake Snowpark** | Data ingestion & processing |
| **Snowflake ML** | Model Registry & warehouse inference |
| **scikit-learn** | Random Forest Classifier |
| **pandas / numpy** | Data manipulation |
| **Python 3.10+** | Core language |

---

## 📌 Key Snowflake SQL Queries

```sql
-- Prediction summary
SELECT
    COUNT(*)                                          AS TOTAL_PREDICTIONS,
    SUM(CASE WHEN PREDICTED_CHURN = 1 THEN 1 END)    AS PREDICTED_CHURN,
    SUM(CASE WHEN PREDICTED_CHURN = 0 THEN 1 END)    AS PREDICTED_NO_CHURN
FROM ML_PROJECT.CHURN.CHURN_PREDICTIONS;

-- Feature importance
SELECT * FROM ML_PROJECT.CHURN.FEATURE_IMPORTANCE ORDER BY IMPORTANCE DESC;

-- Churn summary view
CREATE OR REPLACE VIEW ML_PROJECT.CHURN.CHURN_SUMMARY AS
SELECT
    COUNT(*)                                          AS TOTAL_CUSTOMERS,
    SUM(CASE WHEN PREDICTED_CHURN = 1 THEN 1 END)    AS PREDICTED_CHURN,
    SUM(CASE WHEN PREDICTED_CHURN = 0 THEN 1 END)    AS PREDICTED_NO_CHURN
FROM ML_PROJECT.CHURN.CHURN_PREDICTIONS;
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- IBM Telco Customer Churn dataset
- Snowflake ML documentation
- scikit-learn contributors
