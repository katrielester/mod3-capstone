import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
st.markdown("### Evaluate and visualize your model performance with comprehensive metrics and plots.")

# Sidebar file loading status
st.sidebar.header("Files Loaded")

# Paths to files
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
test_features_path = os.path.join(current_dir, 'test_features.csv')
test_labels_path = os.path.join(current_dir, 'test_labels.csv')

# Load model, features, and labels
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

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report["support"] = df_report["support"].astype(int)
    st.dataframe(df_report.style.background_gradient(cmap="coolwarm", subset=["precision", "recall", "f1-score"]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Confusion Matrix Visualization (Heatmap)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )
    fig_cm.update_layout(xaxis_title="Predicted Labels", yaxis_title="Actual Labels")
    st.plotly_chart(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve and AUC")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"**ROC AUC Score:** {roc_auc:.4f}")
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_disp.plot(ax=ax)
    st.pyplot(fig)

else:
    st.warning("Ensure the model and test data are correctly loaded.")
