import numpy as np
import pandas as pd
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,  train_features_array,train_target_array,test_features_array,test_target_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            x_train, y_train, x_test, y_test = (
                train_features_array,
                train_target_array,
                test_features_array,
                test_target_array
            )

            support_vector_machine = SVC()

            model_report = evaluate_model(x_train, y_train, x_test, y_test, {'SVC': support_vector_machine})
            logging.info(f'Model Report: {model_report}')

            best_model_name = 'SVC'
            best_model_score = model_report[best_model_name]

            logging.info(f'Best Model Found, Model Name: {best_model_name}, Score: {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=support_vector_machine
            )
          
        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)