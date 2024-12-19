import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

# Load the model
with open('gbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

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

if submit_button:
    # Prepare input data for prediction
    data = pd.DataFrame({
        'id': [120],  # Replace with a unique ID
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Preprocessing
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Create new features
    data['age_group'] = pd.cut(data['age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
    data = pd.get_dummies(data, columns=['age_group'], drop_first=True)

    data['hypertension_heart_disease'] = (data['hypertension'] == 1) & (data['heart_disease'] == 1)
    data['high_glucose'] = (data['avg_glucose_level'] > 100).astype(int)

    data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 24.9, 29.9, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    data = pd.get_dummies(data, columns=['bmi_category'], drop_first=True)

    # Ensure missing columns are handled
    missing_columns = set(model.feature_names_) - set(data.columns)
    for col in missing_columns:
        data[col] = 0

    # Ensure the data matches the model's expected features
    data = data[model.feature_names_]

    # Scale numerical columns
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    # Prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    # Output
    if prediction == 1:
        st.markdown("<div class='prediction-box green'>Prediction: Stroke</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box red'>Prediction: No Stroke</div>", unsafe_allow_html=True)
    
    st.write(f"Probability of Stroke: {probability[0]:.2f}")

    # LIME explanation
    train_data = pd.read_csv('train_data_for_lime.csv')
    X_train = train_data[model.feature_names_]
    categorical_features = [
        X_train.columns.get_loc(col)
        for col in ['gender_Male', 'ever_married_Yes', 'work_type_Private', 'Residence_type_Urban', 'smoking_status_smokes']
        if col in X_train.columns
    ]
    
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=model.feature_names_,
        class_names=['No Stroke', 'Stroke'],
        categorical_features=categorical_features,
        mode='classification',
        discretize_continuous=True
    )

    # Generate explanation for a single instance
    instance = data.iloc[0].values
    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba
    )

    explanation_list = exp.as_list()
    explanation_df = pd.DataFrame(explanation_list, columns=['feature', 'weight'])

    # Plot explanation using Streamlit
    fig, ax = plt.subplots(figsize=(7, 6))
    explanation_df = explanation_df.sort_values(by='weight')
    bars = ax.barh(explanation_df['feature'], explanation_df['weight'], color='skyblue', edgecolor='black')

    # Add text annotations for bar values
    for bar in bars:
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            round(bar.get_width(), 2),
            va='center'
        )

    ax.set_xlabel('Contribution to Prediction')
    ax.set_ylabel('Feature')
    ax.set_title('LIME Explanation for Instance')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)
