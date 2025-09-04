import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException
import re
from src.utils import save_object
from scipy.sparse import hstack, csr_matrix

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/preprocessor.pkl')


class SalaryRangeTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        salary_min = []
        salary_max = []
        for salary in X:
            if isinstance(salary, str):
                match = re.match(r'(\d+)[Kk]?\s*-\s(\d+)[Kk]?', salary)
                if match:
                    salary_min.append(int(match.group(1)) * 1000)  # Convert to integer (thousands)
                    salary_max.append(int(match.group(2)) * 1000)
                else:
                    salary_min.append(np.nan)
                    salary_max.append(np.nan)
            else:
                salary_min.append(np.nan)
                salary_max.append(np.nan)

        return pd.DataFrame({'salary_min': salary_min, 'salary_max': salary_max}, index=X.index) 


class DataTransformation:
    def __init__(self):
        self._config = DataTransformationConfig()

    def get_data_transformation_pipeline(self):
        try:
           
            numerical_columns = ['telecommuting', 'has_company_logo', 'has_questions']
            categorical_columns = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type', 'required_education', 'required_experience', 'industry', 'function']
            salary_column = ['salary_range']     

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])  
            salary_pipeline = Pipeline(steps=[
                ('salary_range', SalaryRangeTransformer())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns),
                ('salary_pipeline', salary_pipeline, salary_column)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        try:
            X_train = train_data.drop(columns=['fraudulent'], axis=1)
            y_train = train_data['fraudulent']
            X_test = test_data.drop(columns=['fraudulent'], axis=1)
            y_test = test_data['fraudulent']

            print(f"Columns in X_train before transformation: {X_train.columns}")
            print(f"Missing values in X_train: {X_train.isnull().sum()}")


            X_train['salary_range'] = X_train['salary_range'].fillna('0K - 0K')
            X_test['salary_range'] = X_test['salary_range'].fillna('0K - 0K')


            preprocessor = self.get_data_transformation_pipeline()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            y_train_sparse = csr_matrix(y_train.values).T
            y_test_sparse = csr_matrix(y_test.values).T


            train_arr = hstack([X_train_transformed, y_train_sparse]).tocsr()

            test_arr = hstack([X_test_transformed, y_test_sparse]).tocsr()

            print(f"Concatenated train_arr shape: {train_arr.shape}")
            print(f"Concatenated test_arr shape: {test_arr.shape}")

            preprocessor_path = self._save_preprocessor(preprocessor)


            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            print(f"Error during data transformation: {e}")
            raise CustomException(e, sys)

    def _save_preprocessor(self, preprocessor):
        try:
          
            save_object(file_path=self._config.preprocessor_obj_file_path, obj=preprocessor)
            return self._config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(f"Error saving preprocessor: {e}", sys)
