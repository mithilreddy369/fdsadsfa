import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import lime.lime_tabular

# Load the Gradient Boosting Model
with open('gbm_model.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Load the training data for LIME
train_data = pd.read_csv('train_data_for_lime.csv')

# Define feature names
feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Prepare training data for LIME
X_train = train_data[feature_names]

# Create a LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['no_stroke', 'stroke'],
    mode='classification'
)

# Function to predict using the GBM model
def predict_stroke(features_array):
    return gbm_model.predict(features_array)[0]

# Function to explain with LIME
def explain_with_lime(instance):
    def predict_proba_fn(X):
        return gbm_model.predict_proba(X)

    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_proba_fn
    )

    explanation_list = exp.as_list()
    explanation_df = pd.DataFrame(explanation_list, columns=['feature', 'weight'])

    # Plot customization with Matplotlib (Stacked Bar Chart)
    plt.figure(figsize=(7, 6))
    explanation_df = explanation_df.sort_values(by='weight')
    bars = plt.barh(explanation_df['feature'], explanation_df['weight'], color='skyblue', edgecolor='black')

    for bar in bars:
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            round(bar.get_width(), 2),
            va='center'
        )

    plt.xlabel('Contribution to Prediction')
    plt.ylabel('Feature')
    plt.title('LIME Explanation for Instance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)

# Function to explain with SHAP
def explain_with_shap(instance):
    explainer = shap.Explainer(gbm_model)
    shap_values = explainer(instance)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)

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

    data_mapped = map_data(input_data)

    features = [
        data_mapped['gender'],
        data_mapped['age'],
        data_mapped['hypertension'],
        data_mapped['heart_disease'],
        data_mapped['ever_married'],
        data_mapped['work_type'],
        data_mapped['Residence_type'],
        data_mapped['avg_glucose_level'],
        data_mapped['bmi'],
        data_mapped['smoking_status']
    ]

    features_array = np.array(features).reshape(1, -1)

    # Create a DataFrame with feature names
    features_df = pd.DataFrame(features_array, columns=feature_names)

    # Make predictions
    pred = predict_stroke(features_array)

    if pred is not None:
        st.write("## Predictions")
        color_class = 'green' if pred == 0 else 'red'
        result = 'No Stroke' if pred == 0 else 'Stroke'
        st.markdown(f'<div class="prediction-box {color_class}">{result}</div>', unsafe_allow_html=True)

        # SHAP explanation
        st.write("## SHAP Explanation")
        explain_with_shap(features_df)

        # LIME explanation
        st.write("## LIME Explanation")
        explain_with_lime(features_df.iloc[0].values)
