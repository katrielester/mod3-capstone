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
st.title("Bank Marketing Campaign Model")
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

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Prediction","Confusion Matrix", "Classification Report", "Threshold Adjustment", "ROC Curve"
])

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

    

    with tab1:
        st.subheader("Single Prediction: Deposit or Not")
        st.sidebar.header("Input Customer Features")

        # Function for User Input
        def user_input_features():
            age = st.sidebar.number_input("Age", 18, 99, 30, 1)
            job = st.sidebar.selectbox("Job", ['admin.', 'blue-collar', 'technician', 'services', 
                                               'management', 'retired', 'student', 'self-employed', 
                                               'unemployed', 'housemaid', 'entrepreneur', 'Other'])
            balance = st.sidebar.number_input("Balance", 0.0, 50000.0, 1000.0, 0.01)
            housing = st.sidebar.selectbox("Housing Loan", [0, 1])
            loan = st.sidebar.selectbox("Personal Loan", [0, 1])
            contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone'])
            month = st.sidebar.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            campaign = st.sidebar.number_input("Campaign (Contacts)", 1.0, 50.0, 2.0, 1.0)
            poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", ['success', 'failure', 'other', 'unknown'])
            pdays_binned = st.sidebar.selectbox("Pdays Binned (Last Contact)", 
                                                ['0: never contacted', 
                                                 '1: in the last 7 days', 
                                                 '2: 7 - 30 days ago', 
                                                 '3: more than 30 days ago'])

            # Map pdays_binned to numerical categories
            pdays_mapping = {
                '0: never contacted': 0,
                '1: in the last 7 days': 1,
                '2: 7 - 30 days ago': 2,
                '3: more than 30 days ago': 3
            }
            pdays_binned_value = pdays_mapping[pdays_binned]

            # DataFrame for model input
            data = pd.DataFrame({
                "age": [age],
                "job": [job],
                "balance": [balance],
                "housing": [housing],
                "loan": [loan],
                "contact": [contact],
                "month": [month],
                "campaign": [campaign],
                "poutcome": [poutcome],
                "pdays_binned": [pdays_binned_value]
            })
            return data

        input_df = user_input_features()
        st.write("### Customer Features")
        st.dataframe(input_df)

        # Prediction
        prediction = model.predict(input_df)
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Prediction: **The customer will SUBSCRIBE to the deposit**")
        else:
            st.success("Prediction: **The customer will NOT SUBSCRIBE to the deposit**")

    with tab2:
        st.subheader("Confusion Matrix")
        matrix_view = st.radio("View Confusion Matrix As", ["Counts", "Percentages"], index=0)
        cm_display = cm / cm.sum(axis=1, keepdims=True) * 100 if matrix_view == "Percentages" else cm
        fmt = ".2f" if matrix_view == "Percentages" else "d"
        fig_cm = px.imshow(
            cm_display,
            text_auto=fmt,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": matrix_view},
        )
        fig_cm.update_layout(xaxis_title="Predicted Labels", yaxis_title="Actual Labels")
        st.plotly_chart(fig_cm)

    with tab3:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).T
        st.dataframe(df_report.style.background_gradient(cmap="coolwarm", subset=["precision", "recall", "f1-score"]))

    with tab4:
        st.subheader("Threshold Adjustment")
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, step=0.05)
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm_thresh = confusion_matrix(y_test, y_pred_thresh)
        st.write(f"Metrics at Threshold = {threshold:.2f}:")
        precision_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[0, 1]) if cm_thresh[1, 1] + cm_thresh[0, 1] > 0 else 0
        recall_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[1, 0]) if cm_thresh[1, 1] + cm_thresh[1, 0] > 0 else 0
        st.write(f"- **Precision:** {precision_thresh:.2%}")
        st.write(f"- **Recall:** {recall_thresh:.2%}")
        st.plotly_chart(px.imshow(cm_thresh, text_auto=True, color_continuous_scale="Blues"))

    with tab5:
        st.subheader("ROC Curve")
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.write(f"**ROC AUC Score:** {roc_auc:.4f}")
        roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
        fig, ax = plt.subplots()
        roc_disp.plot(ax=ax)
        st.pyplot(fig)

    st.sidebar.header("Metrics Summary")
    st.sidebar.metric("Precision", f"{precision:.2%}")
    st.sidebar.metric("Recall", f"{recall:.2%}")
    st.sidebar.metric("F1-Score", f"{f1_score:.2%}")
    st.sidebar.metric("Accuracy", f"{accuracy:.2%}")
    
else:
    st.warning("Ensure the model and test data are correctly loaded.")
