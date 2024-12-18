import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import Pool

# Load the model
with open('gbm_model.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Feature names and types from training
cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
all_features = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 
    'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 
    'age_group_Young_Adult', 'age_group_Adult', 'age_group_Middle_Aged', 'age_group_Senior',
    'hypertension_heart_disease', 'high_glucose', 'bmi_category_Normal',
    'bmi_category_Overweight', 'bmi_category_Obese', 'age_hypertension', 
    'bmi_stroke_interaction', 'high_glucose_heart_disease'
]

# Preprocessing function
def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    if 'bmi' in data.columns:
        data['bmi'] = imputer.fit_transform(data[['bmi']])
    else:
        data['bmi'] = 0

    # Encode categorical variables
    data = pd.get_dummies(data, columns=cat_features, drop_first=True)

    # Add missing columns with default values
    for col in all_features:
        if col not in data.columns:
            data[col] = 0

    # Align column order with training data
    data = data[all_features]

    # Scale numerical features
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data

# Streamlit app
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

# Prediction function
if submit_button:
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    data_df = pd.DataFrame([input_data])
    preprocessed_data = preprocess_data(data_df)

    # Create CatBoost Pool for categorical features
    pool = Pool(preprocessed_data, cat_features=[all_features.index(f) for f in cat_features if f in all_features])

    # Make predictions
    prediction = gbm_model.predict(pool)

    # Display results
    st.write("## Predictions")
    color_class = 'green' if prediction[0] == 0 else 'red'
    result = 'No Stroke' if prediction[0] == 0 else 'Stroke'
    st.markdown(f'<div class="prediction-box {color_class}">{result}</div>', unsafe_allow_html=True)
