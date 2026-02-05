
import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Credit Risk", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load('best_credit_risk_model.pkl')
    encoders = {
        'Sex': joblib.load('Sex_encoder.pkl'),
        'Job': joblib.load('Job_encoder.pkl'),
        'Housing': joblib.load('Housing_encoder.pkl'),
        'Saving accounts': joblib.load('Saving_accounts_encoder.pkl'),
        'Checking account': joblib.load('Checking_account_encoder.pkl')
    }
    target_encoder = joblib.load('target_encoder.pkl')
    return model, encoders, target_encoder

model, encoders, target_encoder = load_models()

st.title("Credit Risk Predictor")

# Sidebar
st.sidebar.header("Applicant Details")
age = st.sidebar.slider("Age", 19, 75, 35)
credit_amount = st.sidebar.slider("Credit Amount (€)", 250, 18424, 4000)
duration = st.sidebar.slider("Duration (months)", 4, 72, 18)

sex = st.sidebar.selectbox("Sex", encoders['Sex'].classes_)
job = st.sidebar.selectbox("Job", encoders['Job'].classes_)
housing = st.sidebar.selectbox("Housing", encoders['Housing'].classes_)
saving_accounts = st.sidebar.selectbox("Saving accounts", encoders['Saving accounts'].classes_)
checking_account = st.sidebar.selectbox("Checking account", encoders['Checking account'].classes_)

if st.sidebar.button("PREDICT RISK", type="primary"):
    input_data = {
        'Age': age,
        'Sex': encoders['Sex'].transform([sex])[0],
        'Job': encoders['Job'].transform([job])[0],
        'Housing': encoders['Housing'].transform([housing])[0],
        'Saving accounts': encoders['Saving accounts'].transform([saving_accounts])[0],
        'Checking account': encoders['Checking account'].transform([checking_account])[0],
        'Credit amount': credit_amount,
        'Duration': duration
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    risk = target_encoder.inverse_transform([prediction])[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", risk)
    col2.metric("Low Risk %", f"{probabilities[1]*100:.1f}%")
    col3.metric("High Risk %", f"{probabilities[0]*100:.1f}%")
    
    if risk == 'good':
        st.success("APPROVE - Low Risk!")
        st.markdown(" **CREDIT APPROVED** ")
    else:
        st.error(" **REJECT** - High Risk!")
        st.markdown("**HIGH RISK** ")
        st.markdown("### **Review required** ")
    
    st.subheader("Input Summary")
    summary_df = input_df.copy()
    summary_df['Sex'] = sex
    summary_df['Job'] = job
    summary_df['Housing'] = housing
    summary_df['Saving accounts'] = saving_accounts
    summary_df['Checking account'] = checking_account
    summary_df['Predicted'] = risk
    st.dataframe(summary_df)

st.markdown("---")

