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
    RocCurveDisplay,
)

# App setup
st.title("Model Evaluation and Visualization")
st.sidebar.header("Files Loaded")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
test_features_path = os.path.join(current_dir, 'test_features.csv')
test_labels_path = os.path.join(current_dir, 'test_labels.csv')

# Load files
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

try:
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path).iloc[:, 0]
    st.sidebar.success("Test data loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading test data: {e}")

if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics summary
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    st.sidebar.header("Metrics Summary")
    st.sidebar.metric("Precision", f"{precision:.2%}")
    st.sidebar.metric("Recall", f"{recall:.2%}")
    st.sidebar.metric("F1-Score", f"{f1_score:.2%}")
    st.sidebar.metric("Accuracy", f"{accuracy:.2%}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix","Classification Report","Threshold Adjustment","ROC Curve"])
    with tab1:
        # Toggle between raw counts and percentages
        matrix_view = st.radio("View Confusion Matrix As", ["Counts", "Percentages"], index=0, key="matrix_view")
        if matrix_view == "Percentages":
            cm_display = cm / cm.sum(axis=1, keepdims=True) * 100
            fmt = ".2f"  # Format for percentages
        else:
            cm_display = cm
            fmt = "d"  # Format for raw counts

        fig_cm = px.imshow(
            cm_display,
            text_auto=fmt,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": matrix_view},
        )
        fig_cm.update_layout(xaxis_title="Predicted Labels", yaxis_title="Actual Labels")
        st.plotly_chart(fig_cm, key=f"confusion_matrix_{matrix_view}")
        st.write(
            "The confusion matrix visualizes the model's predictions. "
            "You can switch between raw counts and percentages using the toggle above."
        )
    
    with tab2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).T
        df_report["support"] = df_report["support"].astype(int)
        st.dataframe(df_report.style.background_gradient(cmap="coolwarm", subset=["precision", "recall", "f1-score"]))
        st.write("The classification report includes Precision, Recall, F1-Score, and Support for each class.")

    with tab3:
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, step=0.05)
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm_thresh = confusion_matrix(y_test, y_pred_thresh)
        # Metrics for Adjusted Threshold
        accuracy_thresh = (cm_thresh[0, 0] + cm_thresh[1, 1]) / cm_thresh.sum()
        precision_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[0, 1]) if cm_thresh[1, 1] + cm_thresh[0, 1] > 0 else 0
        recall_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[1, 0]) if cm_thresh[1, 1] + cm_thresh[1, 0] > 0 else 0
        f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if precision_thresh + recall_thresh > 0 else 0
        st.write(f"**Metrics at Threshold = {threshold:.2f}:**")
        st.write(f"- **Precision:** {precision_thresh:.2%}")
        st.write(f"- **Recall:** {recall_thresh:.2%}")
        st.write(f"- **F1-Score:** {f1_thresh:.2%}")
        st.write(f"- **Accuracy:** {accuracy_thresh:.2%}")

        st.write(f"Confusion Matrix for Threshold = {threshold}")
        st.plotly_chart(
            px.imshow(cm_thresh, text_auto=True, color_continuous_scale="Blues"),
            key=f"confusion_matrix_{threshold}",
        )
        

    with tab4:
        st.subheader("ROC Curve")
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.write(f"**ROC AUC Score:** {roc_auc:.4f}")
        roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
        fig, ax = plt.subplots()
        roc_disp.plot(ax=ax)
        st.pyplot(fig)
        st.write(
            "The ROC Curve shows the tradeoff between the True Positive Rate (Recall) "
            "and False Positive Rate at various thresholds."
        )
else:
    st.warning("Ensure the model and test data are correctly loaded.")
