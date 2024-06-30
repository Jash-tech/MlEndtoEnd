import pandas as pd 
import numpy as np 
import sys
import os
import mlflow
import mlflow.sklearn
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


class ModelEvaluation:
    def __init__(self):
        logging.info("Model Evaluation Started")

    def eval_metrics(self,actual, pred):

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return  mae
    
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(np.hstack((train_array[:, :-2], train_array[:, -1:])), np.hstack((test_array[:, :-2], test_array[:, -1:])))

            model_path=os.path.join("artifacts","model.pkl")
             
            model=load_object(model_path)
            # Set the model registry to use the default local filesystem

            mlflow.set_registry_uri("")
            #This line sets the URI (Uniform Resource Identifier) for the MLflow model registry to an empty string. 
            # The model registry URI is used by MLflow to know where to store and retrieve model versions.


            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)
            #mlflow.get_tracking_uri() gets the current tracking URI that MLflow uses for logging and tracking experiments.
           #urlparse(...).scheme parses this URI and extracts the scheme part (the protocol part like http, https, file, etc.).

            with mlflow.start_run():
                prediction=model.predict(X_test)
                mae=self.eval_metrics(y_test,prediction)
                


                
                

                #mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                #mlflow.log_metric("r2", r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                #The provided code snippet decides whether to log the model and register it in the MLflow Model Registry based on the type of tracking URI being used. 
                # If the URI scheme is not "file", it registers the model with the name "ml_model". 
                # If the URI scheme is "file", it simply logs the model without registering it.

        except Exception as e:
            raise CustomException(e,sys)

