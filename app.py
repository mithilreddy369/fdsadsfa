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
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model_and_scaler()

# Streamlit App UI
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

# Prediction Logic
if submit_button:
    # Prepare the data for prediction
    data = pd.DataFrame({
        'id': [120],
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

    # Preprocessing (same as original code)
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    try:
        # 1. Age Category (Age Group)
        data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
        # 2. Glucose Level to BMI Ratio (Health Indicator)
        data['glucose_bmi_ratio'] = data['avg_glucose_level'] / data['bmi']
    
        # 3. Health Risk Score (Hypertension + Heart Disease)
        data['health_risk_score'] = data['hypertension'] + data['heart_disease']
    
        # 4. Work Type and Smoking Status Interaction
        if 'smoking_status' in data.columns:
            data['work_smoking_interaction'] = data['work_type_Private'].astype(int) * data['smoking_status'].map({
                'smokes': 1, 'never smoked': 0, 'formerly smoked': 0, 'Unknown': 0
            }).astype(int)
    
        # 5. Residence and Smoking Interaction (Urban Lifestyle)
        if 'smoking_status' in data.columns:
            data['urban_smoking_interaction'] = data['Residence_type_Urban'].astype(int) * data['smoking_status'].map({
                'smokes': 1, 'never smoked': 0, 'formerly smoked': 0, 'Unknown': 0
            }).astype(int)
    
        # 6. BMI and Age Interaction (Age-Related Obesity Risk)
        data['age_bmi_interaction'] = data['age'] * data['bmi']
    
        # 7. Body Mass Index (BMI) Categorization
        data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 24.9, 29.9, 40, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])
    
        # 8. Interaction between Gender and Smoking Status
        if 'smoking_status' in data.columns:
            data['gender_smoking_interaction'] = data['gender_Male'].astype(int) * data['smoking_status'].map({
                'smokes': 1, 'never smoked': 0, 'formerly smoked': 0, 'Unknown': 0
            }).astype(int)
    
        # 9. Average Glucose Level Category
        data['glucose_level_category'] = pd.cut(data['avg_glucose_level'], bins=[0, 70, 100, 140, 200, 500], labels=['Low', 'Normal', 'High', 'Very High', 'Extremely High'])
    
        # Create dummy variables for categorical columns
        data = pd.get_dummies(data, columns=['age_group', 'glucose_level_category', 'bmi_category'], drop_first=True)

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
    # Ensure missing columns are handled
    # Manually get the feature names from the model (from the booster attribute)
    feature_names = model.get_booster().feature_names

    # Ensure that all columns are in the input data
    missing_columns = set(feature_names) - set(data.columns)
    for col in missing_columns:
        data[col] = 0
    
    
    # Ensure the order of columns matches the model's expected order
    data = data[feature_names]
    
    # Scale numerical columns
    numerical_columns = ['age', 'avg_glucose_level', 'bmi', 'glucose_bmi_ratio', 'age_bmi_interaction']
    data[numerical_columns] = scaler.transform(data[numerical_columns])
    
    # Prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    threshold = 0.001
    thresholded_prediction = (probability >= threshold).astype(int)

    # Display prediction result
    st.markdown("""
        <div class="prediction-box green">
            <h3>Prediction Result</h3>
            <p><strong>Stroke Prediction:</strong> {} (Probability: {:.2f})</p>
        </div>
    """.format("Stroke" if thresholded_prediction[0] == 1 else "No Stroke"), unsafe_allow_html=True)

    # Load training data for LIME (assuming 'train_data_for_lime.csv' exists)
    train_data = pd.read_csv('train_data_for_lime.csv')

    # Define feature names for LIME
    X_train = train_data[feature_names]

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
