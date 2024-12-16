# Bank Marketing Campaign Prediction

## Project Overview
This project predicts customer subscription to term deposits based on a bank's marketing campaign data. The analysis focuses on optimizing **Precision** to minimize resource wasting and false positives, aligning with the bank's goal of cost-efficient marketing.

## Workflow
1. **Data Cleaning**:
   - Addressed missing values, inconsistent formats, and outliers to ensure data quality.
2. **Feature Engineering**:
   - Developed additional features (e.g., customer engagement metrics, demographic categorizations) to enhance predictive power.
3. **Model Testing and Optimization**:
   - Evaluated multiple classification models and optimized the best-performing one using GridSearchCV for hyperparameter tuning.
4. **Evaluation and Recommendations**:
   - Assessed the model's precision, recall, and cost-saving potential to guide actionable strategies.

## Key Results
### Model Performance
- **Best Model**: XGBoost Classifier (tuned via GridSearchCV)
- **Precision**: 0.7838  
- **Recall**: 0.5567  
- **F1-Score**: 0.6511  
- **Accuracy**: 0.7127

### Precision-Driven Insights
1. **High Precision for Subscribed Customers**:
   - With a precision of **78.38%**, nearly **4 out of 5 customers** predicted to subscribe are actual subscribers, significantly reducing resource wastage.
2. **Resource Savings**:
   - The model avoided marketing to **699 non-subscribers**, ensuring targeted and cost-effective campaigns.
3. **Seasonality and Demographics**:
   - Features like **month_oct** and **job_services** were identified as the most influential, providing actionable insights for campaign timing and customer segmentation.

### Feature Importance Highlights
- **Top Features**:
  - `month_oct` (24.8% importance): October emerges as the most favorable period for term deposit campaigns.
  - `job_services` (14.4%): Customers in the service sector are key targets.
  - `poutcome_success` (7.1%): Success in previous campaigns strongly predicts future subscriptions.
- **Financial Indicators**:
  - Features like `balance` and `housing` provide additional predictive context, targeting financially stable customers.

## Recommendations for Future Campaigns
1. **Leverage High-Precision Predictions**:
   - Deploy the model to focus marketing efforts on high-probability subscribers, ensuring efficient resource allocation.
2. **Enhance Features**:
   - Enrich customer data with metrics like income, spending patterns, or engagement history to improve prediction accuracy.
3. **Threshold Optimization**:
   - Experiment with dynamic decision thresholds to adapt to varying campaign goals (e.g., balancing Precision and Recall).

## Future Improvements
1. **Monitor and Retrain**:
   - Implement continuous monitoring to track model performance and retrain periodically to address potential data drift.
2. **Dynamic Campaign Strategies**:
   - Explore the use of real-time data to adjust campaign focus dynamically, ensuring sustained alignment with marketing objectives.
3. **Ensemble Techniques**:
   - Investigate stacking or voting methods to combine strengths of Logistic Regression, LightGBM, and XGBoost for enhanced robustness.

## Files
- **Notebook**: `mod3-capstone.ipynb`
- **Best Model**: `model.pkl`
- **Streamlit**: `stream.py`

## Final Thoughts
The final XGBoost model successfully achieves the objective of predicting term deposit subscriptions with a high level of precision. By minimizing false positives, the bank can effectively target potential subscribers, leading to significant cost savings and improved campaign outcomes. Continuous monitoring, threshold optimization, and iterative enhancements are recommended to sustain and expand the model's effectiveness in dynamic environments.
