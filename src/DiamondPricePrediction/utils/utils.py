

import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)

def load_object(file_path):
    try:
        print(f"Attempting to load object from {file_path}")
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            raise FileNotFoundError(f"File {file_path} not found")

        with open(file_path,'rb') as file_obj:
            obj = pickle.load(file_obj)
            print(f"Successfully loaded object from {file_path}")
            return obj
    except Exception as e:
        print(f"Exception occurred in load_object function: {str(e)}")
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)

