import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
import csv
import os
import time

# File paths
xgb_path = "xgboost_best_model.pkl"
rf_path = "random_forest_best_model.pkl"
log_file = "predictions_log.csv"

# Ensure log file exists
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Model", "Prediction", "Probability", "Fraud_Label"])

# Load models
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)

st.set_page_config(layout="wide")
st.title("🧠 Fraud Detection Prediction Dashboard")

# Split screen
left, right = st.columns(2)

with left:
    st.subheader("🎯 Make a Prediction")

    model_choice = st.selectbox("Choose Model:", ["XGBoost (Optuna-tuned)", "Random Forest"])
    education_level = st.number_input("Education Level", min_value=0.0, max_value=10.0, value=5.0)
    policy_type = st.number_input("Policy Type", min_value=0.0, max_value=5.0, value=1.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    insurance_type = st.number_input("Insurance Type", min_value=0.0, max_value=5.0, value=1.0)
    cause_of_death = st.number_input("Cause of Death", min_value=0.0, max_value=5.0, value=1.0)
    death_location = st.number_input("Death Location", min_value=0.0, max_value=5.0, value=1.0)
    beneficiary_relation = st.number_input("Beneficiary Relation", min_value=0.0, max_value=5.0, value=1.0)
    medical_history = st.number_input("Medical History", min_value=0.0, max_value=5.0, value=1.0)
    treatment_code = st.number_input("Treatment Code", min_value=0.0, max_value=5.0, value=1.0)
    claim_history = st.number_input("Claim History", min_value=0.0, max_value=5.0, value=1.0)
    claim_amount = st.number_input("Claim Amount", min_value=0.0, value=10000.0)
    months_as_customer = st.number_input("Months as Customer", min_value=0, value=10)

    input_data = pd.DataFrame([{
        "education_level": education_level,
        "Policy_Type": policy_type,
        "Gender": 0 if gender == "Male" else 1,
        "Insurance_Type": insurance_type,
        "cause_of_death": cause_of_death,
        "death_location": death_location,
        "beneficiary_relation": beneficiary_relation,
        "medical_history": medical_history,
        "Treatment_Code": treatment_code,
        "Claim History": claim_history,
        "Claim_Amount": claim_amount,
        "Months_as_Customer": months_as_customer
    }])

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            time.sleep(1)

            if model_choice == "XGBoost (Optuna-tuned)":
                model = xgb_model
            else:
                model = rf_model

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            label = "Fraud" if prediction == 1 else "Not Fraud"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, model_choice, prediction, round(prob, 4), label])

            st.success(f"Prediction: {label}")
            st.write(f"**Probability of fraud:** {prob:.2%}")

            st.experimental_rerun()  # <--- Auto-refresh page after prediction

with right:
    st.subheader("📊 Live Dashboard")
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)

        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", len(log_df))
        col2.metric("Fraud Rate", f"{(log_df['Prediction'] == 1).mean() * 100:.1f}%")

        st.markdown("### Fraud Distribution")
        st.bar_chart(log_df["Fraud_Label"].value_counts())

        st.markdown("### Prediction Log")
        st.dataframe(log_df.tail(10), use_container_width=True)
