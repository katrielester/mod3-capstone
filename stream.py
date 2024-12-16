import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# Set up the Streamlit app
st.title("Model Evaluation and Visualization")

# Paths to files
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
test_features_path = os.path.join(current_dir, 'test_features.csv')
test_labels_path = os.path.join(current_dir, 'test_labels.csv')

# Load model, features, and labels
st.sidebar.header("Files Loaded")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

try:
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path).iloc[:, 0]  # Assuming the label column is the first column
    st.sidebar.success("Test data loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading test data: {e}")

if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Display Classification Report
    st.header("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).T)

    # Display Confusion Matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Display ROC Curve
    st.header("ROC Curve and AUC")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"ROC AUC Score: {roc_auc:.4f}")
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    st.pyplot(roc_disp.figure_)

else:
    st.warning("Ensure the model and test data are correctly loaded.")
