import os
import sys
import pandas as pd
from src.DiamondPricePrediction.exception import customexception
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        # Define base paths
        self.artifact_dir = "artifacts"
        self.preprocessor_filename = "preprocessor.pkl"
        self.model_filename = "model.pkl"

        # Construct full paths
        self.preprocessor_path = os.path.join(self.artifact_dir, self.preprocessor_filename)
        self.model_path = os.path.join(self.artifact_dir, self.model_filename)

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required files exist"""
        if not os.path.exists(self.artifact_dir):
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def predict(self, features):
        try:
            logging.info("Starting prediction process...")

            # Load preprocessor
            logging.info(f"Loading preprocessor from {self.preprocessor_path}")
            preprocessor = load_object(self.preprocessor_path)
            logging.info("Preprocessor loaded successfully")

            # Load model
            logging.info(f"Loading model from {self.model_path}")
            model = load_object(self.model_path)
            logging.info("Model loaded successfully")

            # Transform features
            logging.info(f"Transforming features with shape: {features.shape}")
            scaled_data = preprocessor.transform(features)
            logging.info("Features transformed successfully")

            # Make prediction
            logging.info("Making prediction...")
            pred = model.predict(scaled_data)
            logging.info(f"Prediction successful: {pred}")

            return pred

        except Exception as e:
            import traceback
            logging.error(f"Error in prediction: {str(e)}")
            logging.error(traceback.format_exc())
            raise customexception(e, sys)



class CustomData:
    """Class to handle custom input data for diamond price prediction"""

    # Valid values for categorical features
    VALID_CUTS = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    VALID_COLORS = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    VALID_CLARITIES = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        """Initialize with diamond features"""
        # Validate numeric inputs
        self._validate_numeric('carat', carat, min_val=0.1, max_val=10.0)
        self._validate_numeric('depth', depth, min_val=40.0, max_val=80.0)
        self._validate_numeric('table', table, min_val=40.0, max_val=80.0)
        self._validate_numeric('x', x, min_val=0.1, max_val=20.0)
        self._validate_numeric('y', y, min_val=0.1, max_val=20.0)
        self._validate_numeric('z', z, min_val=0.1, max_val=20.0)

        # Validate categorical inputs
        self._validate_categorical('cut', cut, self.VALID_CUTS)
        self._validate_categorical('color', color, self.VALID_COLORS)
        self._validate_categorical('clarity', clarity, self.VALID_CLARITIES)

        # Set attributes
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

        logging.info("CustomData object created with validated inputs")

    def _validate_numeric(self, name, value, min_val=None, max_val=None):
        """Validate numeric inputs"""
        try:
            value = float(value)
            if min_val is not None and value < min_val:
                raise ValueError(f"{name} must be at least {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{name} must be at most {max_val}")
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be a valid number")

    def _validate_categorical(self, name, value, valid_values):
        """Validate categorical inputs"""
        if value not in valid_values:
            raise ValueError(f"{name} must be one of {valid_values}")

    def get_data_as_dataframe(self):
        """Convert the data to a pandas DataFrame"""
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f'DataFrame created with shape: {df.shape}')
            return df
        except Exception as e:
            logging.error(f'Exception occurred in get_data_as_dataframe: {str(e)}')
            raise customexception(e, sys)