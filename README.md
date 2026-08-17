# Telco Customer Churn — Classification Models & Streamlit App

## a. Problem Statement

Customer churn — when a subscriber stops using a company's service — is one of
the most expensive problems in the telecom industry, since acquiring a new
customer costs far more than retaining an existing one. This project builds
and compares five classification models that predict whether a telecom
customer will churn based on their account details, services subscribed, and
billing information, then exposes the best-performing models through an
interactive Streamlit web app so predictions and evaluation metrics can be
explored without touching code.

## b. Dataset Description

- **Source:** IBM Telco Customer Churn dataset (publicly available, also
  mirrored on Kaggle as "Telco Customer Churn").
- **Instances:** 7,043 customers
- **Features:** 19 predictive features after dropping the `customerID`
  identifier — demographics (gender, senior citizen status, partner,
  dependents), account information (tenure, contract type, payment method,
  billing charges), and subscribed services (phone, internet, online
  security, tech support, streaming, etc.)
- **Target:** `Churn` — binary (`Yes` / `No`, encoded as 1 / 0)
- **Preprocessing:** `TotalCharges` had a handful of blank values for
  brand-new customers (tenure = 0); these were coerced to numeric and filled
  with the column median. Categorical features were one-hot encoded and
  numeric features were standardized, both inside an sklearn `Pipeline` so
  the exact same transformation is applied at inference time in the
  Streamlit app. An engineered `tenure_group` feature (New / Established /
  Loyal, bucketed from raw tenure in months) was also added, since churn
  risk is known to concentrate heavily among newer customers. Models were
  trained on a 75/25 stratified train/test split (`random_state=7`).

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO URL HERE AFTER YOU PUSH>`

## d. Models Used

Five models were trained on an 80/20 stratified train/test split of the same
dataset, and evaluated with Accuracy, AUC, Precision, Recall, F1 Score, and
Matthews Correlation Coefficient (MCC).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8052 | 0.8579 | 0.6615 | 0.5439 | 0.5969 | 0.4740 |
| Decision Tree | 0.7995 | 0.8343 | 0.6462 | 0.5396 | 0.5881 | 0.4603 |
| kNN | 0.7888 | 0.8128 | 0.6097 | 0.5653 | 0.5867 | 0.4456 |
| Naive Bayes | 0.7161 | 0.8305 | 0.4803 | 0.8630 | 0.6172 | 0.4649 |
| Random Forest (Ensemble) | 0.8177 | 0.8609 | 0.6984 | 0.5503 | 0.6156 | 0.5044 |

*(Regenerate this table by running `model/train_models.py` — it writes the
same numbers to `model/comparison_results.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, close second overall. With regularization (`C=0.5`) it stays well-calibrated and interpretable, and churn drivers here (contract length, tenure, monthly charges) are largely linear/monotonic, which suits it well. |
| Decision Tree | Once depth-limited (`max_depth=6`, `min_samples_leaf=20`) to prevent overfitting, it performs respectably and closely tracks Logistic Regression, showing the earlier weak performance of a single unpruned tree was mostly an overfitting artifact. |
| kNN | Middling performance even with distance-weighted voting and a larger neighborhood (`n_neighbors=11`). Distance-based methods remain sensitive to the high-dimensional one-hot-encoded feature space (curse of dimensionality). |
| Naive Bayes | Lowest precision but by far the highest recall (0.863) — it over-flags customers as "at risk," trading precision for catching almost all true churners. Useful if the business would rather over-flag churn risk than miss it, despite violating its own feature-independence assumption. |
| Random Forest (Ensemble) | Best overall performer — highest Accuracy, AUC, Precision, and MCC. Bagging over many depth/leaf-constrained trees (`n_estimators=150`, `max_depth=10`) reduces the overfitting a single tree suffers from while still capturing non-linear feature interactions the linear model can't. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — best Accuracy, AUC, and MCC (the most reliable single summary metric for imbalanced binary classification), at the cost of being less directly interpretable than Logistic Regression. |

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split used for demo/evaluation
│-- data/
│   └── Telco-Customer-Churn.csv
│-- model/
│   ├── train_models.py     # trains all 5 models, saves .joblib + metrics
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   ├── comparison_results.csv
│   └── comparison_results.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional — .joblib files are already included
streamlit run app.py
```

## Streamlit App Features

- CSV upload for test data (or use the bundled `test_data.csv`)
- Dropdown to select which of the 5 trained models to use
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap and full classification report
- Optional side-by-side comparison of all 5 models on the uploaded data

## Live Links

- **GitHub Repository:** `<PASTE LINK>`
- **Live Streamlit App:** `<PASTE LINK>`
