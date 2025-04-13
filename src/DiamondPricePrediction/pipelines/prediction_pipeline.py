import os
import sys
import pandas as pd
from src.DiamondPricePrediction.exception import customexception
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            print("Starting prediction process...")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            print(f"Loading preprocessor from {preprocessor_path}")
            preprocessor=load_object(preprocessor_path)
            print("Preprocessor loaded successfully")

            print(f"Loading model from {model_path}")
            model=load_object(model_path)
            print("Model loaded successfully")

            print("Transforming features...")
            scaled_data=preprocessor.transform(features)
            print("Features transformed successfully")

            print("Making prediction...")
            pred=model.predict(scaled_data)
            print(f"Prediction successful: {pred}")

            return pred

        except Exception as e:
            import traceback
            print(f"Error in prediction: {str(e)}")
            print(traceback.format_exc())
            raise customexception(e,sys)



class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):

        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity


    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'carat':[self.carat],
                    'depth':[self.depth],
                    'table':[self.table],
                    'x':[self.x],
                    'y':[self.y],
                    'z':[self.z],
                    'cut':[self.cut],
                    'color':[self.color],
                    'clarity':[self.clarity]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)