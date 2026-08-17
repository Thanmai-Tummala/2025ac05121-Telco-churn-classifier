"""
train_models.py

Loads the Telco Customer Churn dataset, preprocesses it, trains five
classification models, evaluates each on a held-out test set, and saves
the fitted pipelines (+ the test split) so the Streamlit app can reuse
them without retraining.

Run:
    python train_models.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 7  # my chosen seed
DATA_PATH = "../data/Telco-Customer-Churn.csv"
TEST_SIZE = 0.25


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # customerID is just an identifier, not a predictive feature
    df = df.drop(columns=["customerID"])

    # TotalCharges is read as object because a few rows have blank strings
    # for brand-new customers (tenure == 0); coerce and fill with median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary target: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Engineered feature: bucket raw tenure (months) into lifecycle stages.
    # Churn risk tends to be highest for very new customers, so giving the
    # model an explicit "New/Established/Loyal" category alongside the raw
    # number can help tree-based models split on it directly.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 48, 100],
        labels=["New", "Established", "Loyal"],
    )

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=SEED),
        # capped depth so the tree can't just memorize the training set
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=SEED
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    churn_data = load_and_clean_data(DATA_PATH)
    features = churn_data.drop(columns=["Churn"])
    target = churn_data["Churn"]

    feats_train, feats_test, target_train, target_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED, stratify=target
    )

    preprocessor = build_preprocessor(features)
    results = {}

    for model_name, estimator in get_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(feats_train, target_train)

        preds = pipeline.predict(feats_test)
        pred_proba = pipeline.predict_proba(feats_test)[:, 1]

        results[model_name] = evaluate(target_test, preds, pred_proba)

        # filename-safe version of the model name
        fname = model_name.lower().replace(" ", "_")
        joblib.dump(pipeline, f"{fname}.joblib")
        print(f"Saved {fname}.joblib  ->  {results[model_name]}")

    # Save the comparison table for the README / app
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "ML Model Name"
    comparison_df.to_csv("comparison_results.csv")
    print("\nComparison table:\n", comparison_df)

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save a test-data CSV (features + true label) for the Streamlit app's
    # upload feature. Kept small so it stays within Streamlit free-tier limits.
    holdout_sample = feats_test.copy()
    holdout_sample["Churn"] = target_test.values
    holdout_sample.to_csv("../test_data.csv", index=False)
    print(f"\nSaved test_data.csv with {len(holdout_sample)} rows")


if __name__ == "__main__":
    main()
