import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)



    except Exception as e:
        logging.info('Exception Occured in save_object function utils')
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        result={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            y_pred=model.predict(X_test)

            mae=round(mean_absolute_error(y_test,y_pred),8)
            result[list(models.keys())[i]] =  mae
        return result


    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)