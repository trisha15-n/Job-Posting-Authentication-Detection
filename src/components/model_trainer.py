import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from scipy.sparse import csr_matrix
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            if isinstance(y_train, csr_matrix):
                y_train = y_train.toarray().flatten()  # Convert to dense array
            if isinstance(y_test, csr_matrix):
                y_test = y_test.toarray().flatten()

            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train) 
            X_test = imputer.transform(X_test)     

            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Random Forest model accuracy: {accuracy}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=rf_model
            )
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)
