from src.DiamondPricePrediction.components.data_ingestion import DataIngestion

from src.DiamondPricePrediction.components.data_transformation import DataTransformation

from src.DiamondPricePrediction.components.model_trainer import ModelTrainer

#from src.DiamondPricePrediction.components.model_evaluation import ModelEvaluation


import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
import pandas as pd

'''
class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            raise customexception(e,sys)
        
    def start_data_transformation(self,train_data_path,test_data_path):
        
        try:
            data_transformation = DataTransformation()
            train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        except Exception as e:
            raise customexception(e,sys)
    
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_trainer.initate_model_training(train_arr,test_arr)
        except Exception as e:
            raise customexception(e,sys)
                
    def start_trainig(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)
        except Exception as e:
            raise customexception(e,sys)
        

'''

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)


