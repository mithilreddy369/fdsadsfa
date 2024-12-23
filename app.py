import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap

# Load the trained model
try:
    loaded_model = joblib.load('gbm_model.pkl')
except FileNotFoundError:
    st.error("Model file 'gbm_model.pkl' not found. Please upload the model file.")
    st.stop()


# Load the training data for LIME explainer (replace with actual path if needed)
try:
    train_data = pd.read_csv('train_data_for_lime.csv')
    feature_names = train_data.drop('stroke', axis=1).columns
except FileNotFoundError:
    st.error("Training data file 'train_data_for_lime.csv' not found.")
    st.stop()

# Create a LIME explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(train_data.drop('stroke', axis=1)),
    feature_names=feature_names,
    class_names=['0', '1'],
    mode='classification'
)


def predict_stroke(input_data):
    # Feature Engineering (needs to match the training data)
    input_data['Risk_Score'] = (
        input_data['age'] +
        (input_data['hypertension'] * 10) +
        (input_data['heart_disease'] * 15) +
        (input_data['avg_glucose_level'] / 10) +
        (input_data['bmi'] / 5)
    )
    data['High_BMI'] = (data['bmi'] > 30).astype(int)
    data['Age_Hypertension_Interaction'] = data['age'] * data['hypertension']
    data['Age_Glucose_Interaction'] = data['age'] * data['avg_glucose_level']
    data['Heart_Disease_Flag'] = (data['heart_disease'] == 1).astype(int)
    data['Hypertension_Flag'] = (data['hypertension'] == 1).astype(int)
    data['High_Stroke_Risk_Flag'] = ((data['age'] > 60) & (data['hypertension'] == 1) & (data['heart_disease'] == 1)).astype(int)
    data['Critical_Stroke_Risk_Flag'] = ((data['age'] > 65) & (data['hypertension'] == 1) & (data['heart_disease'] == 1) & (data['avg_glucose_level'] > 160)).astype(int)
    data['Stroke_Risk_Hypertension_HeartDisease_Glucose'] = ((data['hypertension'] == 1) & (data['heart_disease'] == 1) & (data['avg_glucose_level'] > 140)).astype(int)
    data['Very_High_Diabetes_Risk_Flag'] = ((data['age'] > 65) & (data['bmi'] > 35) & (data['avg_glucose_level'] > 160)).astype(int)

    # One-hot encode categorical columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    input_data = pd.get_dummies(input_data, columns=categorical_cols)

    # Align columns with training data
    missing_cols = set(train_data.columns) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0
    input_data = input_data[train_data.columns.drop('stroke')]

    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data)[:, 1]
    return prediction[0], probability[0]


# Streamlit app
st.title("Stroke Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, value=50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])



# Create input DataFrame
input_data = pd.DataFrame({
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


if st.button("Predict"):
    prediction, probability = predict_stroke(input_data.copy())
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability of Stroke: {probability:.4f}")


    # LIME explanation
    explanation = explainer.explain_instance(
        data_row=np.array(input_data.iloc[0]),  
        predict_fn=loaded_model.predict_proba,
        num_features=len(feature_names)
    )
    explanation.as_pyplot_figure(label=1)
    st.pyplot(plt) #display LIME plot using streamlit
