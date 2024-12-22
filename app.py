import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import numpy as np

# Load the pre-trained GBM model
with open('gbm_model.pkl', 'rb') as f:
    loaded_model = joblib.load(f)

# Load training data for LIME
X_train_final = pd.read_csv('train_data_for_lime.csv')  # Replace with the actual path if necessary

# Sample Input Data (replace with Streamlit input fields)
data = pd.DataFrame({
    'gender': ['Male'],
    'age': [23],
    'hypertension': [0],
    'heart_disease': [0],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [100.51],
    'bmi': [20.3],
    'smoking_status': ['smokes'],
})

# Streamlit app
st.title("Stroke Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=23)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.51)
bmi = st.number_input("BMI", min_value=0.0, value=20.3)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Update the data DataFrame with user inputs
data = pd.DataFrame({
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

# Preprocessing and Feature Engineering (same as before)
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

data['Risk_Score'] = (
    data['age'] +
    (data['hypertension'] * 10) +
    (data['heart_disease'] * 15) +
    (data['avg_glucose_level'] / 10) +
    (data['bmi'] / 5)
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

# Ensure all feature columns are present in the input data
training_feature_names = loaded_model.feature_names_in_
data_X = data.reindex(columns=training_feature_names, fill_value=0)

# Make prediction
if st.button("Predict"):
    prediction = loaded_model.predict(data_X)[0]
    probability = loaded_model.predict_proba(data_X)[:, 1][0]

    st.write(f"Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
    st.write(f"Probability of Stroke: {probability:.4f}")

    # Choose an instance from the test set to explain
    instance_idx = st.slider('Select Instance Index', 0, len(X_train_final) - 1, 0)
    instance = X_train_final.iloc[instance_idx]
    
    # Create LimeTabularExplainer object
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_final.values,  # Training data
        feature_names=X_train_final.columns.tolist(),  # Feature names
        class_names=['No Stroke', 'Stroke'],  # Class names
        discretize_continuous=True
    )
    
    # Explain the instance's prediction
    explanation = explainer.explain_instance(
        instance.values, loaded_model.predict_proba, num_features=len(X_train_final.columns)
    )
    
    # 1. Contribution Features Bar Chart
    st.subheader('1. Feature Contributions Bar Chart')
    fig = explanation.as_pyplot_figure()
    plt.title('Feature Contributions for Stroke Prediction (LIME)')
    st.pyplot(fig)
    
    # 2. Cumulative Feature Importance
    st.subheader('2. Cumulative Feature Importance')
    feature_importance = explanation.as_list()
    feature_names = [feature[0] for feature in feature_importance]
    feature_contributions = [abs(feature[1]) for feature in feature_importance]
    
    cumulative_importance = np.cumsum(feature_contributions)
    
    plt.figure(figsize=(10, 6))
    plt.plot(feature_names, cumulative_importance, marker='o')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance (LIME)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    
    # 3. Pie Chart (Top 5 features and the rest as "Others")
    st.subheader('3. Top 5 Features Contributions (Pie Chart)')
    
    top_n = 5
    top_features = feature_names[:top_n]
    top_contributions = feature_contributions[:top_n]
    other_contributions = sum(feature_contributions[top_n:])
    top_features.append('Others')
    top_contributions.append(other_contributions)
    
    plt.figure(figsize=(8, 8))
    plt.pie(top_contributions, labels=top_features, autopct='%1.1f%%', startangle=90)
    plt.title('Top Features Contributions (LIME)')
    plt.axis('equal')
    st.pyplot(plt)
