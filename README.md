# 📉 Customer Churn Prediction using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishiraghu11/Customer_Churn_Prediction_using_ML/blob/main/Customer_Churn_Prediction_using_ML.ipynb)

---

## 📖 Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in the telecom industry. Retaining an existing customer is significantly cheaper than acquiring a new one, making early churn detection a high-value business problem.

This project builds an **end-to-end churn prediction pipeline** on the IBM Telco Customer Churn dataset. It covers data cleaning, exploratory analysis, class imbalance handling with SMOTE, multi-model training, and a deployable prediction system with per-customer risk scoring and explainability.

---

## 🎯 Objectives

- Identify behavioral and contractual patterns that drive customer churn.
- Handle class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
- Train and compare **three ML models** to select the best performer.
- Output **churn probability** rather than just binary Yes/No predictions.
- Segment customers into **Low / Medium / High churn risk** tiers.
- Explain predictions at the **individual customer level** using feature importances.
- Save the trained model and encoders as `.pkl` files for deployment.

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── Customer_Churn_Prediction_using_ML.ipynb   # Main notebook
├── Telco-Customer-Churn.csv                   # Dataset                     
└── README.md
```

---

## 📊 Dataset

**Source:** IBM Telco Customer Churn Dataset  
**File:** `Telco-Customer-Churn.csv`  
**Records:** 7,043 customers | **Features:** 20 + 1 target (`Churn`)

| Feature Group | Columns |
|---------------|---------|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Charges | `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (Yes / No) |

**Data quality notes:**
- `customerID` dropped (non-predictive identifier).
- 11 rows had whitespace in `TotalCharges` — replaced with `0.0` and cast to float.
- `MonthlyCharges` comma-decimal format normalized to float.
- Class imbalance: ~73% No Churn vs ~27% Churn — addressed with SMOTE.

---

## 🔍 Methodology

### 1. Exploratory Data Analysis
- Histograms with mean/median overlays for `tenure`, `MonthlyCharges`, `TotalCharges`.
- Box plots to assess distribution spread and outliers.
- Correlation heatmap across numerical features.
- Count plots for all categorical features.

### 2. Data Preprocessing
- Label encoding applied to all categorical columns using `sklearn.LabelEncoder`.
- Encoders saved to `encoders.pkl` for consistent reuse during prediction.
- `Churn` encoded as 1 (Yes) / 0 (No).

### 3. Class Imbalance — SMOTE
- Stratified 80/20 train-test split applied **before** SMOTE to prevent data leakage.
- SMOTE applied only on training data to synthetically balance the minority class.
- Post-SMOTE training set achieves a balanced 50/50 class distribution.

### 4. Model Training & Selection
Three models trained with default hyperparameters and evaluated via **5-fold cross-validation**:

| Model | Result |
|-------|--------|
| Decision Tree | Baseline |
| **Random Forest** | **Best CV Accuracy ✅** |
| XGBoost | Competitive |

**Random Forest** selected as the final model.

### 5. Model Evaluation
Final model evaluated on the held-out test set using accuracy score, confusion matrix, and a full classification report (precision, recall, F1 per class).

### 6. Predictive System
Model and encoders loaded from `.pkl` files. New customer data is encoded using saved encoders and passed to the model, returning both a **binary prediction** and a **churn probability score**.

### 7. Risk Segmentation

| Risk Tier | Churn Probability |
|-----------|------------------|
| 🟢 Low Risk | < 30% |
| 🟡 Medium Risk | 30% – 60% |
| 🔴 High Risk | > 60% |

### 8. Individual Explainability
Per-customer feature impact computed as `feature_value × feature_importance`, surfacing the top drivers behind each individual prediction.

---

## 📈 Key Insights

- **High monthly and total charges** are the strongest predictors of churn.
- **Short tenure** significantly elevates churn risk — early-stage customers are the most vulnerable.
- **Month-to-month contracts** are associated with much higher churn than 1- or 2-year contracts.
- **Electronic check** payment method shows a moderate association with churn.
- **Demographics** (gender, senior citizen status) have minimal predictive impact.

---

## 💡 Business Recommendations

- Proactively target **High Risk** customers with personalized retention offers — discounts, contract upgrades, or service bundles.
- Launch **early-tenure onboarding programs** to reduce churn in the first 3–6 months.
- Incentivize month-to-month customers to **switch to annual contracts**.
- Investigate **electronic check users** — payment friction may be an operational churn signal.
- Use **churn probability scores** (not just binary output) to prioritize the retention team's outreach queue efficiently.

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas, NumPy** | Data manipulation and cleaning |
| **Matplotlib, Seaborn** | EDA visualizations |
| **Scikit-learn** | Preprocessing, model training, evaluation |
| **XGBoost** | Gradient boosting classifier |
| **imbalanced-learn** | SMOTE for class imbalance |
| **Pickle** | Model and encoder serialization |
| **Google Colab** | Development environment |

---

## ⚙️ How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

**2. Open the notebook** in Google Colab or Jupyter and run all cells top to bottom.

**3. Predict for a new customer:**
```python
import pickle, pandas as pd

input_data = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 1, 'PhoneService': 'No',
    'MultipleLines': 'No phone service', 'InternetService': 'DSL',
    'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85, 'TotalCharges': 29.85
}

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

df_input = pd.DataFrame([input_data])
for col, enc in encoders.items():
    df_input[col] = enc.transform(df_input[col])

prob = model_data["model"].predict_proba(df_input)[0][1]
prediction = model_data["model"].predict(df_input)[0]
print(f"Prediction : {'Churn' if prediction == 1 else 'No Churn'}")
print(f"Churn Probability: {prob:.2%}")
```

---

## 🚧 Limitations & Future Scope

- Model trained on static historical data — may drift as customer behavior evolves over time.
- Customer support interactions, NPS scores, and real-time usage patterns are not included.
- Hyperparameter tuning (Grid Search / Optuna) could further improve model performance.
- Future work: deploy as a **REST API** (Flask / FastAPI) or integrate into a CRM dashboard for real-time churn scoring.

---

## 👨‍💻 Author

**Rishi Raj Singh Raghuvanshi**
