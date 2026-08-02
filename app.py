import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification - ML Assignment 2")
st.write("Upload test_data.csv, select a model, and view metrics, confusion matrix, and predictions.")

model_options = {
    "Logistic Regression": "models/Logistic_Regression.pkl",
    "Decision Tree": "models/Decision_Tree.pkl",
    "KNN": "models/KNN.pkl",
    "Naive Bayes": "models/Naive_Bayes.pkl",
    "Random Forest": "models/Random_Forest.pkl"
}

selected_model = st.selectbox("Select Machine Learning Model", list(model_options.keys()))
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(data.head())

    target_col = "income" if "income" in data.columns else data.columns[-1]
    X_test = data.drop(columns=[target_col])
    y_test = data[target_col].astype(str).apply(lambda value: 1 if ">50K" in value else 0)

    model = joblib.load(model_options[selected_model])
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["classifier"], "predict_proba") else y_pred

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [
            accuracy_score(y_test, y_pred),
            roc_auc_score(y_test, y_score),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            matthews_corrcoef(y_test, y_pred)
        ]
    })

    st.subheader("Evaluation Metrics")
    st.dataframe(metrics_df)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig)

    st.subheader("Prediction Output")
    output_df = X_test.copy()
    output_df["Actual"] = y_test.map({0: "<=50K", 1: ">50K"})
    output_df["Predicted"] = pd.Series(y_pred).map({0: "<=50K", 1: ">50K"})
    st.dataframe(output_df.head(50))
else:
    st.info("Please upload test_data.csv generated from the notebook.")
