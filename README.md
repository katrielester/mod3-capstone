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
- **Precision**: 0.8063  
- **Recall**: 0.5781  
- **Cost Efficiency**:
  - The model achieves **81% precision**, meaning **81 out of 100 predicted subscribers are actual subscribers**, significantly reducing resource wastage.
  - Avoided marketing to **710 non-subscribers**, optimizing resource allocation.

### Precision-Driven Insights
1. **High Precision for Subscribed Customers**:
   - The model effectively identifies potential subscribers, minimizing unnecessary marketing costs while focusing on high-probability customers.
2. **Resource Savings**:
   - The model reduces false positives to just **104**, ensuring targeted and cost-effective campaigns.

## Recommendations for Future Campaigns
1. **Leverage High-Precision Predictions**:
   - Concentrate marketing efforts on customers predicted to subscribe with high confidence, ensuring better resource utilization.
2. **Enhance Features**:
   - Enrich customer data with metrics like income, spending patterns, or engagement history to improve prediction accuracy.
3. **Threshold Optimization**:
   - Experiment with dynamic decision thresholds to adapt to varying campaign goals (e.g., balancing Precision and Recall).

## Future Improvements
1. **Monitor and Retrain**:
   - Implement continuous monitoring to track model performance and retrain periodically to address potential data drift.
2. **Dynamic Campaign Strategies**:
   - Explore the use of real-time data to adjust campaign focus dynamically, ensuring sustained alignment with marketing objectives.

## Files
- **Notebook**: `mod3-capstone.ipynb`
- **Best Model**: `model.pkl`

## Final Thoughts
The final model successfully achieves the objective of predicting term deposit subscriptions with a high level of Precision. By reducing false positives, the bank can focus resources effectively, leading to significant cost savings and improved campaign outcomes. Continuous monitoring and iterative enhancements are recommended to maintain and expand its effectiveness in dynamic environments.
