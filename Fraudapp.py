import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import os
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
    
    # Inputs for single prediction (same as before)
    # (Add your input fields here like education_level, policy_type, etc.)

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
            st.rerun()

with right:
    st.subheader("📊 Upload Claims CSV for Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV with Claims Data", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        claims_data = pd.read_csv(uploaded_file)
        
        st.write("Uploaded Claims Data:")
        st.dataframe(claims_data.head())  # Display the first few rows of the uploaded file

        # Preprocess data for prediction (ensure the columns match with your model input)
        claims_data["Gender"] = claims_data["Gender"].map({"Male": 0, "Female": 1})
        
        # Assuming the CSV file contains columns that match the model's input
        prediction_data = claims_data[[
            "education_level", "Policy_Type", "Gender", "Insurance_Type", 
            "cause_of_death", "death_location", "beneficiary_relation", 
            "medical_history", "Treatment_Code", "Claim History", 
            "Claim_Amount", "Months_as_Customer"
        ]]
        
        # Make predictions on the whole dataset
        if st.button("🔍 Predict All Claims"):
            with st.spinner("Making batch predictions..."):
                time.sleep(1)
                
                if model_choice == "XGBoost (Optuna-tuned)":
                    model = xgb_model
                else:
                    model = rf_model
                
                predictions = model.predict(prediction_data)
                prob_predictions = model.predict_proba(prediction_data)[:, 1]  # Fraud probabilities
                
                claims_data["Prediction"] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]
                claims_data["Probability"] = prob_predictions

                st.write("Prediction Results:")
                st.dataframe(claims_data)  # Display the predictions

                # Optionally, save results to a new CSV file
                output_file = "predicted_claims.csv"
                claims_data.to_csv(output_file, index=False)
                st.download_button("Download Predicted Claims", output_file)

