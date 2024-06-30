import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from dataclasses import dataclass
from pathlib import Path
from src.utils import save_object,evaluate_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_path=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            X_train, y_train, X_test, y_test = (
                np.hstack((train_array[:, :-2], train_array[:, -1:])),
                train_array[:,-2],
                np.hstack((test_array[:, :-2], test_array[:, -1:])),
                test_array[:,-2]
                 )
            
            models={
            'DTR':DecisionTreeRegressor(min_samples_split=5),'RFR':RandomForestRegressor(),'LR':LinearRegression()
             }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            print(model_report)

            best_model_score = min(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , MAE : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , MAE : {best_model_score}')

            save_object(
                 file_path=self.model_path.trained_model_file_path,
                 obj=best_model
            )


        except Exception as e:
            raise CustomException(e,sys)