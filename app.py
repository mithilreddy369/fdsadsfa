import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define input fields for the Streamlit app
st.title("Stroke Prediction App")

# Collect user input
st.header("Input Features")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Preprocess the input
input_data = {
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi
}

# Encode categorical variables
categorical_data = {
    'gender': [1 if gender == "Male" else 0],
    'ever_married': [1 if ever_married == "Yes" else 0],
    'work_type': [work_type],
    'Residence_type': [1 if residence_type == "Urban" else 0],
    'smoking_status': [smoking_status]
}

# Convert to DataFrame
data = pd.DataFrame(input_data, index=[0])
data = pd.concat([data, pd.DataFrame(categorical_data)], axis=1)

# Feature engineering
data['age_group'] = pd.cut([age], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior']).astype(str)
data['high_glucose'] = (data['avg_glucose_level'] > 100).astype(int)
data['bmi_category'] = pd.cut([bmi], bins=[0, 18.5, 24.9, 29.9, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese']).astype(str)

# One-hot encode categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_category']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Ensure all required columns are present in the same order as training
required_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    # Add all feature-engineered and one-hot encoded columns used during training
]
data = data.reindex(columns=required_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Stroke! Probability: {probability:.2f}")
    else:
        st.success(f"Low Risk of Stroke. Probability: {probability:.2f}")
