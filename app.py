import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained GBM model
with open("gbm_model.pkl", "rb") as file:
    gbm_model = pickle.load(file)

# Streamlit App
st.title("Stroke Prediction App")

# Input Form
st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox(
        "Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", ["never smoked", "formerly smoked", "smokes"]
    )

    data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status,
    }
    return pd.DataFrame(data, index=[0])

# Collect input features
data = user_input_features()

# Preprocessing
def preprocess_data(data):
    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Adding missing columns if necessary
    expected_columns = [
        "gender_Male", "gender_Other", "ever_married_Yes",
        "work_type_Never_worked", "work_type_Private",
        "work_type_Self-employed", "work_type_children",
        "Residence_type_Urban", "smoking_status_formerly smoked",
        "smoking_status_never smoked", "smoking_status_smokes"
    ]
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Scale numerical features
    numerical_columns = ["age", "avg_glucose_level", "bmi"]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data

# Preprocess input data
preprocessed_data = preprocess_data(data)

# Make predictions
prediction = gbm_model.predict(preprocessed_data)
prediction_proba = gbm_model.predict_proba(preprocessed_data)[:, 1]

# Display results
st.subheader("Prediction")
result = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"
st.write(result)

st.subheader("Prediction Probability")
st.write(f"Probability of Stroke: {prediction_proba[0]:.2f}")
