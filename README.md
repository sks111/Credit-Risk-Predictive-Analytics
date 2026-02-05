# 💳 Credit Risk Predictive Analytics  
### End-to-End Machine Learning Project for Loan Default Risk Assessment
![App link](https://credit-risk-predictive-analytics-io9eu3i3vgynhrqrkl8gpw.streamlit.app/)
![Python](https://img.shields.io/badge/Python-Machine%20Learning-3776AB)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Modeling-F7931E)
![Domain](https://img.shields.io/badge/Domain-Credit%20Risk-blue)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## 📌 Project Overview

This project focuses on building a **Credit Risk Predictive Analytics system** using **machine learning** to classify loan applicants as **low risk** or **high risk**.  
The solution is based on historical credit data and is designed to support **data-driven lending decisions** in financial institutions.

The project demonstrates a **complete real-world ML workflow**, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and categorical encoding
- Training and evaluating multiple ML algorithms
- Hyperparameter tuning
- Model selection and persistence
- Deployment using a lightweight application

---

## 🎯 Business Problem

Incorrect credit approval decisions can result in:
- High loan default rates
- Financial losses
- Poor credit portfolio quality

### Objective
To build a predictive model that:
- Accurately identifies **high-risk applicants**
- Reduces false approvals
- Supports consistent and objective loan decisions

---

## 🧠 Dataset Description

The project uses the **German Credit Dataset**, which contains customer demographic, financial, and loan-related attributes.

### Feature Groups
- **Demographic:** Age, Sex, Job  
- **Financial:** Checking account, Saving accounts  
- **Loan-related:** Credit amount, Duration, Purpose  
- **Housing:** Living situation  
- **Target Variable:** Credit Risk (Good / Bad)

---

## 🛠️ Tools & Technologies

| Category | Tools |
|--------|------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Environment | Jupyter Notebook |
| Deployment | Flask / Streamlit |
| Serialization | Pickle |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand:
- Dataset structure and quality
- Feature distributions
- Credit risk concentration across customer segments
- Relationships between financial variables and default risk

### Key Insights from EDA
- Financial attributes (checking and saving account status) strongly influence credit risk
- Larger loan amounts and longer durations increase default probability
- Credit risk is unevenly distributed across customer groups

These insights guided **feature preprocessing and model selection**.

---

## 🧩 Data Preprocessing & Feature Engineering

The following preprocessing steps were applied:
- Handling missing values
- Encoding categorical variables using label encoders
- Encoding the target variable for binary classification
- Saving encoders to ensure consistent preprocessing during inference

### Saved Encoders
- `Sex_encoder.pkl`
- `Job_encoder.pkl`
- `Housing_encoder.pkl`
- `Checking_account_encoder.pkl`
- `Saving_accounts_encoder.pkl`
- `target_encoder.pkl`

---

## 🤖 Machine Learning Models Used

Multiple classification algorithms were trained and evaluated using the same train–test split.  
Hyperparameter tuning was performed using **GridSearchCV**.

### Models Implemented
1. **Decision Tree Classifier (DT)**
2. **Random Forest Classifier (RF)**
3. **Extra Trees Classifier (ET)**
4. **XGBoost Classifier (XGB)**

---

## 📈 Model Evaluation & Accuracy

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Risk-sensitive comparison

📌 In credit risk problems, **recall for high-risk cases is more important than raw accuracy**, as approving a risky applicant can lead to financial loss.

### Accuracy Results (from Notebook)

| Model | Accuracy | Tuned Parameters |
|-----|---------|------------------|
| **Decision Tree** | **0.581** | `max_depth=5`, `min_samples_leaf=1`, `min_samples_split=2` |
| **Random Forest** | **0.619** | `n_estimators=100`, `min_samples_split=10`, `min_samples_leaf=2`, `max_depth=None` |
| **Extra Trees** | **0.648** | `n_estimators=100`, `max_depth=10`, `min_samples_leaf=2`, `min_samples_split=5` |
| **XGBoost** | **0.648** | Gradient boosting with tuned parameters |

---

## 🏆 Final Model Selection

Based on **overall accuracy, robustness, and ensemble stability**, the **Extra Trees Classifier and XGBoost** achieved the **highest accuracy (0.648)**.

The final selected model was saved as:

best_credit_risk_model.pkl


This model provides a **balanced trade-off between performance and generalization**, making it suitable for real-world credit risk assessment.

---

## 🧠 How the Model Works

1. Applicant data is provided as input  
2. Input features are encoded using saved encoders  
3. Encoded data is passed to the trained ML model  
4. Model outputs:
   - **Low Credit Risk**
   - **High Credit Risk**
5. Prediction acts as a **decision-support tool** for loan approval

---

## 🚀 Model Deployment

A lightweight application (`app.py`) was built to demonstrate real-time predictions.

### Deployment Flow
- Accept user input
- Apply preprocessing using saved encoders
- Load trained model
- Return credit risk prediction

This simulates how the model can be integrated into a real lending system.

---
## OUTPUT

![](https://github.com/sks111/Credit-Risk-Predictive-Analytics/blob/main/output.png)

