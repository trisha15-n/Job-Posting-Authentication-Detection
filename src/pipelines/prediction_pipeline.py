import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix
from src.exception import CustomException
from src.logger import logging
import joblib  

class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')

    def load_model(self):
        """Load the trained model from disk."""
        try:
            logging.info("Loading the trained model from disk")
            model = joblib.load(self.model_path)  
            return model
        except Exception as e:
            raise CustomException(f"Error loading model: {e}", sys)

    def load_preprocessor(self):
        """Load the preprocessor object from disk."""
        try:
            logging.info("Loading the preprocessor object from disk")
            preprocessor = joblib.load(self.preprocessor_path)  
            return preprocessor
        except Exception as e:
            raise CustomException(f"Error loading preprocessor: {e}", sys)

    def preprocess_data(self, input_data: pd.DataFrame):
        """Preprocess the input data to be ready for model prediction."""
        try:
            logging.info(f"Initial shape of input data: {input_data.shape}")
            logging.info(f"Initial NaN counts in input data: {input_data.isna().sum()}")

            if 'fraudulent' in input_data.columns:
                input_data = input_data.drop(columns=['fraudulent'])

            input_data['salary_range'] = input_data['salary_range'].fillna('0K - 0K')


            empty_columns = input_data.columns[input_data.isna().all()].tolist()
            if empty_columns:
                logging.info(f"Dropping columns with all NaN values: {empty_columns}")
                input_data = input_data.drop(columns=empty_columns)

            logging.info(f"Shape after dropping empty columns: {input_data.shape}")
            logging.info(f"NaN counts after dropping empty columns: {input_data.isna().sum()}")

            numeric_cols = input_data.select_dtypes(include=[np.number]).columns
            categorical_cols = input_data.select_dtypes(exclude=[np.number]).columns

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

   
            numeric_imputer = SimpleImputer(strategy='mean')
            input_data[numeric_cols] = numeric_imputer.fit_transform(input_data[numeric_cols])


            categorical_imputer = SimpleImputer(strategy='most_frequent')
            input_data[categorical_cols] = categorical_imputer.fit_transform(input_data[categorical_cols])

            logging.info(f"Shape after imputation: {input_data.shape}")
            logging.info(f"NaN counts after imputation: {input_data.isna().sum()}")

            
            logging.info(f"Data before preprocessor: {input_data.head()}")

  
            preprocessor = self.load_preprocessor()

            if hasattr(preprocessor, 'feature_names_in_'):
                expected_cols = list(preprocessor.feature_names_in_)
           
                for col in expected_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
           
                input_data = input_data[expected_cols]
                logging.info(f"Aligned input columns to preprocessor: {list(input_data.columns)}")
            else:
                logging.warning("Preprocessor does not have feature_names_in_. Skipping column alignment.")

     
            if hasattr(preprocessor, 'feature_names_in_'):
                logging.info(f"Preprocessor expects columns: {list(preprocessor.feature_names_in_)}")
            logging.info(f"Input data columns before transform: {list(input_data.columns)}")

       
            X_transformed = preprocessor.transform(input_data)
            logging.info(f"Shape after preprocessor.transform: {X_transformed.shape}")

   
            if isinstance(X_transformed, csr_matrix):
                X_transformed = X_transformed.toarray()

  
            model = None
            model_n_features = None
            try:
                model = self.load_model()
                if hasattr(model, 'n_features_in_'):
                    model_n_features = model.n_features_in_
                    logging.info(f"Model expects n_features_in_: {model_n_features}")
                elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'n_features_in_'):
                    model_n_features = model.estimators_[0].n_features_in_
                    logging.info(f"Model expects n_features_in_: {model_n_features}")
            except Exception as e:
                logging.warning(f"Could not load model for feature count logging: {e}")

            logging.info(f"Shape of transformed data: {X_transformed.shape}")
            nan_count = np.isnan(X_transformed).sum()
            logging.info(f"NaN counts after transformation: {nan_count}")

     
            if nan_count > 0:
                X_transformed = np.nan_to_num(X_transformed, nan=0)
                logging.info(f"NaNs found after transformation. Filled with 0. New NaN count: {np.isnan(X_transformed).sum()}")

            if model_n_features is not None:
                n_transformed = X_transformed.shape[1]
                if n_transformed > model_n_features:
                    logging.warning(f"Trimming {n_transformed - model_n_features} extra features to match model input.")
                    X_transformed = X_transformed[:, :model_n_features]
                elif n_transformed < model_n_features:
                    logging.warning(f"Padding {model_n_features - n_transformed} missing features with zeros to match model input.")
                    pad_width = model_n_features - n_transformed
                    X_transformed = np.pad(X_transformed, ((0,0),(0,pad_width)), 'constant')
                logging.info(f"Shape after alignment to model: {X_transformed.shape}")

            return X_transformed
        except Exception as e:
            raise CustomException(f"Error during data preprocessing: {e}", sys)

    def make_prediction(self, input_data: pd.DataFrame):
        try:
            X_processed = self.preprocess_data(input_data)
            model = self.load_model()
            predictions = model.predict(X_processed)
            return predictions
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)

if __name__ == "__main__":
    try:
        pipeline = PredictionPipeline()
        test_data = pd.DataFrame({
            'job_id': [1234],
            'title': ['Data Scientist'],
            'location': ['New York'],
            'department': ['Engineering'],
            'salary_range': ['100K - 120K'],
            'company_profile': ['A great company'],
            'description': ['Analyze data and build models'],
            'requirements': ['Python, ML, Statistics'],
            'benefits': ['Health insurance, 401k'],
            'telecommuting': [1],
            'has_company_logo': [1],
            'has_questions': [0],
            'employment_type': ['Full-time'],
            'required_experience': ['Mid'],
            'required_education': ['Bachelor'],
            'industry': ['Tech'],
            'function': ['Data Science']
        })
        predictions = pipeline.make_prediction(test_data)
        print("Predictions:", predictions)
    except Exception as e:
        print(f"Error in prediction process: {e}")
        logging.error(f"Error in prediction process: {e}")
