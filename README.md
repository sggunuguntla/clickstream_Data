# Customer Conversion Analysis Using Clickstream Data

## Project Overview
This project analyzes clickstream data to predict customer conversion, estimate revenue, and segment users based on browsing behavior. An interactive Streamlit app is developed for real-time insights.

## Objectives
- Predict whether a customer will make a purchase
- Estimate potential revenue
- Segment customers for targeted marketing

## Dataset
Source: UCI Machine Learning Repository  
Files:
- train.csv
- test.csv

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Streamlit

## Machine Learning Models
- Classification: Random Forest
- Regression: Gradient Boosting Regressor
- Clustering: K-Means

## Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: RMSE, MAE, R²
- Clustering: Silhouette Score

## How to Run

bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
