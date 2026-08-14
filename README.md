# Credit Card Fraud Detection

## Overview 
An end-to-end Machine Learning pipeline utilizing XGBoost to identify fraudulent credit card transactions in heavily imbalanced datasets. This project includes dataset preprocessing, feature engineering (such as spatial distance via the Haversine formula), class imbalance handling using `scale_pos_weight`, evaluation, and model serialization.

![Deployment Overview](deployment.png)

## Project Structure
```
├── Credit_Card_Fraud_Detection.ipynb   # Main Jupyter Notebook
├── Credit_Card_Fraud_Detection.pkl     # Trained XGBoost Model File
├── app.py                              # Web App / Deployment Script
├── templates/
│   └── index.html                      # UI Template for deployment
├── README.md                           # Documentation
└── deployment.png                      # Screenshot of UI
```

## Summary
- PR-AUC: 0.9151
- Fraud Class Recall: 0.95 (Captures ~95% of actual frauds)
- Overall Accuracy: 0.99

## Author
Ujjwal Kumar
GitHub: [https://github.com/ujjwalkumar14b](https://github.com/ujjwalkumar14b)
