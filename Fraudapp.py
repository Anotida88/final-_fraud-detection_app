import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# File paths for models
xgb_path = "xgboost_best_model.pkl"
rf_path = "random_forest_best_model.pkl"

# Load models
xgb_model = joblib.load(xgb_path)
rf_model = joblib.load(rf_path)

# Title of the app
st.title("Fraud Detection Prediction App")
st.write("Select model, enter claim details, and get a fraud prediction.")

# Model selection dropdown
model_choice = st.selectbox("Choose Model:", ["XGBoost (Optuna-tuned)", "Random Forest"])

# Input fields for claim details
st.subheader("Enter Claim Information")

education_level = st.number_input("Education Level", min_value=0.0, max_value=10.0, value=5.0)
policy_type = st.number_input("Policy Type", min_value=0.0, max_value=5.0, value=1.0)
gender = st.selectbox("Gender", ["Male", "Female"])  # Gender input
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

# Preprocess Gender (convert to numeric)
input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})

# Real-time prediction with a progress bar
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        time.sleep(2)  # Simulate model prediction delay

        # Prediction logic based on selected model
        if model_choice == "XGBoost (Optuna-tuned)":
            model = xgb_model
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
        else:
            model = rf_model
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

        # Show result
        st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
        st.write(f"**Probability of fraud:** {prob:.2%}")

        # Feature importance visualization (XGBoost)
        if model_choice == "XGBoost (Optuna-tuned)":
            st.subheader("Feature Importance (XGBoost)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            shap.summary_plot(shap_values, input_data)
            st.pyplot(plt)

        # Display confusion matrix for model evaluation (use actual data and predictions for this)
        st.subheader("Confusion Matrix (Model Evaluation)")
        y_true = [1, 0, 1, 0, 1, 0]  # Replace with actual values
        y_pred = [1, 0, 1, 1, 0, 0]  # Replace with model predictions
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # Add a pie chart for fraud vs non-fraud prediction distribution
        st.subheader("Fraud vs Non-Fraud Prediction Distribution")
        labels = ["Fraud", "Not Fraud"]
        sizes = [prob * 100, (1 - prob) * 100]
        colors = ['#ff9999', '#66b3ff']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
