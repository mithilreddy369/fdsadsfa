import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import shap

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open('gbm_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Streamlit App UI
st.title("Stroke Prediction and Explanation App")

# Input fields for new data
st.sidebar.header("Input Features")

# Define the input fields
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=54)
hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.sidebar.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', value=144.51)
bmi = st.sidebar.number_input('BMI', value=32.3)
smoking_status = st.sidebar.selectbox('Smoking Status', ['smokes', 'never smoked', 'formerly smoked'])

# Prepare the data for prediction
data = pd.DataFrame({
    'id': [120],
    'gender': [gender],
    'age': [age],
    'hypertension': [1 if hypertension == 'Yes' else 0],
    'heart_disease': [1 if heart_disease == 'Yes' else 0],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Preprocessing (same as original code)
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

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

# Display prediction result
st.subheader("Prediction Result")
st.write("Prediction (1 indicates stroke, 0 indicates no stroke):", prediction[0])
st.write("Probability of stroke:", probability[0])

# Load training data for LIME (assuming 'train_data_for_lime.csv' exists)
train_data = pd.read_csv('train_data_for_lime.csv')

# Define feature names for LIME
X_train = train_data[model.feature_names_]

# Identify categorical feature indices for LIME
categorical_features = [
    X_train.columns.get_loc(col) 
    for col in ['gender_Male', 'ever_married_Yes', 'work_type_Private', 'Residence_type_Urban', 'smoking_status_smokes']
    if col in X_train.columns
]

# Initialize LimeTabularExplainer
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

# Extract and visualize explanations
explanation_list = exp.as_list()
explanation_df = pd.DataFrame(explanation_list, columns=['feature', 'weight'])

# Plot the explanation using a horizontal bar chart
st.subheader("LIME Explanation")
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

# Sort the explanation by weight to visualize the cumulative importance
explanation_df_sorted = explanation_df.sort_values(by='weight', ascending=False)
explanation_df_sorted['cumulative_weight'] = explanation_df_sorted['weight'].cumsum()

# Plot cumulative feature importance
st.subheader("Cumulative Feature Importance")
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.plot(explanation_df_sorted['feature'], explanation_df_sorted['cumulative_weight'], marker='o', color='teal')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Cumulative Contribution to Prediction')
ax2.set_title('Cumulative Feature Importance')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xticklabels(explanation_df_sorted['feature'], rotation=90)
st.pyplot(fig2)

# Prepare data for pie chart
positive_weights = explanation_df[explanation_df['weight'] > 0]
positive_weights_sum = positive_weights['weight'].sum()

# Calculate percentage contribution
positive_weights['percentage'] = (positive_weights['weight'] / positive_weights_sum) * 100

# Plot pie chart
st.subheader("Feature Contribution to Positive Prediction")
fig3, ax3 = plt.subplots(figsize=(7, 7))
ax3.pie(
    positive_weights['percentage'],
    labels=positive_weights['feature'],
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab20.colors
)
ax3.set_title('Feature Contribution to Positive Prediction')
st.pyplot(fig3)
