import streamlit as st
import pandas as pd
import joblib
import os

# File paths
xgb_path= "xgboost_best_model.pkl"
rf_path = "random_forest_best_model.pkl"

rf_path = "C:/Users/taonga/Downloads/project/random_forest_best_model.pkl"
# Load models
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)

# Title
st.title("Fraud Detection Prediction App")
st.write("Select model, enter claim details, and get a fraud prediction.")

# Model selection
model_choice = st.selectbox("Choose Model:", ["XGBoost (Optuna-tuned)", "Random Forest"])

# Input fields
st.subheader("Enter Claim Information")

education_level = st.number_input("Education Level", min_value=0.0, max_value=10.0, value=5.0)
policy_type = st.number_input("Policy Type", min_value=0.0, max_value=5.0, value=1.0)
gender = st.selectbox("Gender", ["0", "1"])  # Assume 0 and 1 for Gender
insurance_type = st.number_input("Insurance Type", min_value=0.0, max_value=5.0, value=1.0)

cause_of_death = st.number_input("Cause of Death", min_value=0.0, max_value=5.0, value=1.0)
death_location = st.number_input("Death Location", min_value=0.0, max_value=5.0, value=1.0)
beneficiary_relation = st.number_input("Beneficiary Relation", min_value=0.0, max_value=5.0, value=1.0)

medical_history = st.number_input("Medical History", min_value=0.0, max_value=5.0, value=1.0)
treatment_code = st.number_input("Treatment Code", min_value=0.0, max_value=5.0, value=1.0)

claim_history = st.number_input("Claim History", min_value=0.0, max_value=5.0, value=1.0)
claim_amount = st.number_input("Claim Amount", min_value=0.0, value=10000.0)
months_as_customer = st.number_input("Months as Customer", min_value=0, value=10)

# Create input DataFrame
input_data = pd.DataFrame([{
    "education_level": education_level,
    "Policy_Type": policy_type,
    "Gender": gender,
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

# Predict
if st.button("Predict"):
    if model_choice == "XGBoost (Optuna-tuned)":
        model = xgb_model
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
    else:
        model = rf_model
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    # Show result
    st.markdown(f"### Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
    st.markdown(f"**Probability of fraud:** {prob:.2%}")
