import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline

# Example input data (replace with your actual test data)
test_data = pd.DataFrame({
    'job_id': [1],
    'title': ['Data Scientist'],
    'location': ['San Francisco'],
    'department': ['AI'],
    'salary_range': ['100K - 120K'],
    'company_profile': ['Tech Company'],
    'description': ['Job description...'],
    'requirements': ['Python, ML'],
    'benefits': ['Health Insurance'],
    'telecommuting': [1],
    'has_company_logo': [1],
    'has_questions': [1],
    'employment_type': ['Full-time'],
    'required_experience': ['3-5 years'],
    'required_education': ['Master\'s'],
    'industry': ['Technology'],
    'function': ['Development']
})

# Instantiate the prediction pipeline and make predictions
prediction_pipeline = PredictionPipeline()

# Make prediction
predictions = prediction_pipeline.make_prediction(test_data)

# Output predictions
print("Predictions:", predictions)
