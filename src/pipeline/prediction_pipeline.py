import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to the preprocessor and model files
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load the preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the model
            predictions = model.predict(data_scaled)
            return predictions

        except Exception as e:
            logging.info("Exception occurred in the prediction pipeline")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, length, diameter, height, whole_weight, 
                 shucked_weight, viscera_weight, shell_weight, rings):
        self.length = length
        self.diameter = diameter
        self.height = height
        self.whole_weight = whole_weight
        self.shucked_weight = shucked_weight
        self.viscera_weight = viscera_weight
        self.shell_weight = shell_weight
        self.rings = rings

    def get_data_as_dataframe(self):
        try:
            # Creating a dictionary from the input features
            custom_data_input_dict = {
                'Length': [self.length],
                'Diameter': [self.diameter],
                'Height': [self.height],
                'Whole weight': [self.whole_weight],
                'Shucked weight': [self.shucked_weight],
                'Viscera weight': [self.viscera_weight],
                'Shell weight': [self.shell_weight],
                'Rings': [self.rings]
            }

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in data preparation')
            raise CustomException(e, sys)
