# 📊 Customer Churn Analysis & Prediction using Machine Learning

## 📌 Project Overview

Customer churn is a critical business problem where retaining existing customers is often more cost-effective than acquiring new ones.
This project implements an end-to-end churn analysis and prediction pipeline using Python, focusing on interpretability and business usability rather than complex black-box models.

The project combines data analysis, feature engineering, and machine learning to identify churn drivers and predict churn probability for individual customers.

## 🎯 Objectives

Analyze customer behavior and identify churn patterns

Build an interpretable machine learning model to predict churn

Generate probability-based churn risk instead of only binary predictions

Explain churn predictions at an individual customer level

Translate analytical insights into actionable business recommendations

## 🛠️ Tools & Technologies

Python

Pandas, NumPy – data manipulation

Matplotlib, Seaborn – data visualization

Scikit-learn – machine learning (Logistic Regression)

Google Colab

## 🔍 Methodology

### 1️⃣ Data Cleaning & EDA

Handled missing and inconsistent values

Converted categorical variables into numerical format

Performed exploratory data analysis to understand churn trends across:

Contract type

Tenure

Monthly and total charges

Service usage

### 2️⃣ Feature Engineering

Removed non-predictive identifiers

Encoded categorical features

Prepared a clean, ML-ready dataset

### 3️⃣ Machine Learning Model

Logistic Regression was used due to its interpretability and suitability for churn prediction

Dataset was split using stratified train-test split

Model evaluation focused on:

Recall (Churn = 1)

ROC-AUC score

### 4️⃣ Probability-Based Prediction

Model outputs churn probabilities instead of only Yes/No predictions

Customers are categorized into:

Low Risk

Medium Risk

High Risk of churn

### 5️⃣ Explainable ML

Feature coefficient analysis identifies global churn drivers

Individual customer-level impact analysis explains why a specific churn prediction was made

## 📈 Key Insights

High monthly and total charges are the strongest drivers of churn

Shorter tenure significantly increases churn risk

Long-term contracts reduce churn probability

Demographic features such as gender and senior citizen status have minimal impact

## 💡 Business Recommendations

Target high-risk customers with discounts or contract upgrades

Focus retention efforts on early-tenure customers

Encourage bundled services for customers with high monthly charges

Use churn probability thresholds to prioritize retention actions

## 🚧 Limitations & Future Scope

Model is trained on historical data and may not capture recent behavior changes

Customer support interactions and sentiment data are not included

Future work may include real-time churn prediction or deployment as an API

## 🏁 Conclusion

This project demonstrates how data analysis and interpretable machine learning can be combined to solve a real-world business problem.
By focusing on churn probability and explainability, the solution provides actionable insights that can support effective customer retention strategies.
