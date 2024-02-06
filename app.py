import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("loan_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Features used during training
model_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                  'Credit_History', 'TotalIncome', 'Gender_Male', 'Gender_Female', 'Married_Yes', 'Married_No',
                  'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_Yes', 'Self_Employed_No',
                  'Property_Area_Urban', 'Property_Area_Rural', 'Property_Area_Semiurban']

# Function to preprocess user input
def preprocess_input(gender, married, dependents, education, self_employed, applicant_income,
                     coapplicant_income, total_income, loan_amount, loan_amount_term, credit_history, property_area):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'TotalIncome': [total_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
    })

    # Perform one-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data)

    # Align input features with model features
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    return input_data

# Streamlit app
def main():
    st.title("Loan Eligibility Prediction")

    # User input form
    st.sidebar.header("User Input")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=1)
    total_income = st.sidebar.number_input("Total Income", min_value=0, step=1)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1)
    loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0, step=1)
    credit_history = st.sidebar.selectbox("Credit History", [0, 1])
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Preprocess user input
    processed_input = preprocess_input(gender, married, dependents, education, self_employed,
                                       applicant_income, coapplicant_income, total_income, loan_amount,
                                       loan_amount_term, credit_history, property_area)

    # Make predictions
    if st.sidebar.button("Predict Loan Eligibility"):
        prediction = model.predict(processed_input)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        st.sidebar.success(f"The loan application is {result}")

if __name__ == "__main__":
    main()
