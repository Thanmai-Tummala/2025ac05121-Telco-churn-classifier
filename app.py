"""
app.py — Streamlit demo app for the Telco Customer Churn classification assignment.

Lets a user upload a CSV of test data, pick one of five trained models,
and see the model's evaluation metrics, predictions, and a confusion
matrix / classification report on the uploaded data.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest.joblib",
}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


st.title("📊 Telco Customer Churn — Classification Model Explorer")
st.write(
    "Upload a test CSV (same columns as the training data, including the "
    "`Churn` label) to evaluate any of the five trained models."
)

# --- Sidebar controls ---------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=False)

# --- Load data ------------------------------------------------------------
df = None
if use_sample:
    df = pd.read_csv("test_data.csv")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if df is None:
    st.info("Upload a CSV from the sidebar, or check 'Use bundled test_data.csv' to try the demo.")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head())

if "Churn" not in df.columns:
    st.error("The uploaded CSV must include a 'Churn' column (0/1) with true labels to compute metrics.")
    st.stop()

X = df.drop(columns=["Churn"])
y_true = df["Churn"]

# --- Predict --------------------------------------------------------------
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# --- Metrics ---------------------------------------------------------------
st.subheader(f"Evaluation metrics — {model_name}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

# --- Confusion matrix + classification report ------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

# --- Compare all models (optional extra) -----------------------------------
st.subheader("Compare all models on this data")
if st.checkbox("Run all 5 models on the uploaded data"):
    rows = []
    for name, path in MODEL_FILES.items():
        m = load_model(path)
        pred = m.predict(X)
        proba = m.predict_proba(X)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "AUC": roc_auc_score(y_true, proba),
            "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred),
            "MCC": matthews_corrcoef(y_true, pred),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))
