import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_monitoring import ModelEvaluation


# obj=DataIngestion()

# train_data_path,test_data_path=obj.initiate_dataingestion()

# data_transformation=DataTransformation()

# train_arr,test_arr=data_transformation.initiate_datatransformation(train_data_path,test_data_path)


# model_trainer_obj=ModelTrainer()
# model_trainer_obj.initiate_model_training(train_arr,test_arr)

# model_evaluation_obj=ModelEvaluation()
# model_evaluation_obj.initiate_model_evaluation(train_arr,test_arr)


class Training:
    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion...")
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
            logging.info("Completed data ingestion")
            return train_data_path,test_data_path
        except Exception as e:
            logging.info("Error in data Ingestion")
            raise CustomException(e,sys)
        

    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Starting data Transformation...")
            data_transformation=DataTransformation()
            train_arr,test_arr=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            logging.info("Completed data transformation...")
            return train_arr,test_arr
        
        except Exception as e:
            logging.info("Error in Data transformation...")
            raise CustomException(e,sys)
        
    def start_model_training(self,train_arr,test_arr):
        try:
            logging.info("Starting model training...")
            model_trainer=ModelTrainer()
            model_trainer.initiate_model_training(train_arr,test_arr)
            
        except Exception as e:
            logging.info("Error in model training...")
            raise CustomException(e,sys)
        
    def start_training(self):
        try:
            logging.info("Compiling all components")
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)
            logging.info("Completed...")

        except Exception as e:
            logging.info("Error in Model overall")
            raise CustomException(e,sys)
            


# if __name__ == "__main__":
#     try:
#         training_obj = Training()
#         training_obj.start_training()
#     except Exception as e:
#         raise CustomException(e,sys)






        