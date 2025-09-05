import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline

st.title("Job Posting Authentication Prediction")

feature_names = [
    'job_id', 'title', 'location', 'department', 'salary_range', 'company_profile',
    'description', 'requirements', 'benefits', 'telecommuting', 'has_company_logo',
    'has_questions', 'employment_type', 'required_experience', 'required_education',
    'industry', 'function'
]

input_data = {}
for feature in feature_names:

    if feature in ['job_id', 'telecommuting', 'has_company_logo', 'has_questions']:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0)
    else:
        input_data[feature] = st.text_input(f"Enter {feature}")

if st.button("Predict"):
    try:
 
        df = pd.DataFrame([input_data])
        pipeline = PredictionPipeline()
        prediction = pipeline.make_prediction(df)
        label = int(prediction[0])
        if label == 1:
            st.success("Prediction: Fraudulent job posting detected.")
        else:
            st.success("Prediction: Legitimate job posting.")
    except Exception as e:
        st.error(f"Error: {e}")
