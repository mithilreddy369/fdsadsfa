import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the model
with open('gbm_model1.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Data preprocessing functions
def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    if 'bmi' in data.columns:
        data['bmi'] = imputer.fit_transform(data[['bmi']])
    else:
        data['bmi'] = 0  # Add a default value if 'bmi' is missing

    # Ensure all expected categorical columns are present
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_columns:
        if col not in data.columns:
            data[col] = None

    # Encode categorical variables
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Feature engineering
    if 'age' in data.columns:
        data['age_group'] = pd.cut(data['age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
        data = pd.get_dummies(data, columns=['age_group'], drop_first=True)
    else:
        data['age'] = 0  # Add a default age if missing

    data['hypertension_heart_disease'] = (data['hypertension'] == 1) & (data['heart_disease'] == 1)
    data['high_glucose'] = (data['avg_glucose_level'] > 100).astype(int)
    
    if 'bmi' in data.columns:
        data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 24.9, 29.9, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        data = pd.get_dummies(data, columns=['bmi_category'], drop_first=True)

    # Additional derived features
    data['age_hypertension'] = (data['age'] > 50) & (data['hypertension'] == 1)
    if 'stroke' in data.columns:
        data['bmi_stroke_interaction'] = ((data.get('bmi_category_Overweight', 0) == 1) | 
                                          (data.get('bmi_category_Obese', 0) == 1)) & (data['stroke'] == 1)
    else:
        data['bmi_stroke_interaction'] = 0

    data['high_glucose_heart_disease'] = (data['high_glucose'] == 1) & (data['heart_disease'] == 1)

    # Scale numerical features
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    for col in numerical_columns:
        if col not in data.columns:
            data[col] = 0  # Add default values for missing numerical columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data

# Streamlit app
st.markdown("""
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        .input-group { margin-bottom: 15px; }
        .prediction-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .green { background-color: #28a745; color: white; }
        .red { background-color: #dc3545; color: white; }
        .prediction-row { display: flex; justify-content: space-around; }
    </style>
""", unsafe_allow_html=True)

st.title('Brain Stroke Prediction App')

# Input form
with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col2:
        age = st.slider('Age', min_value=0, max_value=100, value=20)
    with col3:
        hypertension = st.selectbox('Hypertension', [0, 1])

    col4, col5, col6 = st.columns(3)
    with col4:
        heart_disease = st.selectbox('Heart Disease', [0, 1])
    with col5:
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    with col6:
        work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children'])

    col7, col8, col9 = st.columns(3)
    with col7:
        residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
    with col8:
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=80.13)
    with col9:
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=23.4)

    col10, col11 = st.columns(2)
    with col10:
        smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
    
    submit_button = st.form_submit_button(label='Predict')

# Map categorical values to numerical values
def map_data(data):
    return {
        'gender': 0 if data['gender'] == 'Male' else 1,
        'age': data['age'],
        'hypertension': data['hypertension'],
        'heart_disease': data['heart_disease'],
        'ever_married': 1 if data['ever_married'] == 'Yes' else 0,
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self_employed': 3, 'children': 4}[data['work_type']],
        'Residence_type': 0 if data['residence_type'] == 'Rural' else 1,
        'avg_glucose_level': data['avg_glucose_level'],
        'bmi': data['bmi'],
        'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}[data['smoking_status']]
    }

if submit_button:
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    data_df = pd.DataFrame([input_data])
    preprocessed_data = preprocess_data(data_df)

    # Make predictions
    prediction = gbm_model.predict(preprocessed_data)

    st.write("## Predictions")
    color_class = 'green' if prediction[0] == 0 else 'red'
    result = 'No Stroke' if prediction[0] == 0 else 'Stroke'
    st.markdown(f'<div class="prediction-box {color_class}">{result}</div>', unsafe_allow_html=True)

    st.write("Prediction complete with preprocessed features.")
