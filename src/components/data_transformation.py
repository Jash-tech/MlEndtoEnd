import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_ingestion import DataIngestion

from dataclasses import dataclass
from pathlib import Path
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from src.components.model_training import ModelTrainer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data trnsformation Process of creation preprocessor started")
            cat_cols=['Power', 'Brand', 'Gas', 'Region']
            num_cols=['ClaimNb', 'Exposure', 'CarAge', 'DriverAge', 'Density', 'ClaimFreq']

            num_pipeline=Pipeline (
                    steps=[
                        ("imputer",SimpleImputer()),
                        ("scaler",MinMaxScaler())
                    ]
                )
            
            power_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder", OrdinalEncoder()),
                ("scaler",MinMaxScaler())
            ])

            # Define the preprocessing steps for 'Brand', 'Gas', 'Region' columns
            other_cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ("scaler",MinMaxScaler())
            ])

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("power_pipeline",power_pipeline,['Power']),
                    ("other_cat_pipeline",other_cat_pipeline,['Brand', 'Gas', 'Region'])
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation Begins")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading Train and Test data Completed")


            preprocessing_obj=self.get_data_transformation()

            logging.info("Creating X and Y")
            target_column='ClaimAmount'

            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            input_feature_test_df=test_df.drop(columns=target_column,axis=1)

            target_feature_train_df=train_df[target_column]
            target_feature_test_df=test_df[target_column]

            input_feature_train_df_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_df_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_path,
                        obj=preprocessing_obj)

            logging.info("preprocessing pickle file saved")

            return (train_arr,test_arr)




        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    a=DataIngestion()
    train_path,test_path=a.initiate_data_ingestion()
    b=DataTransformation()
    train_ar,test_ar=b.initiate_data_transformation(train_path,test_path)
    c=ModelTrainer()
    c.initiate_model_training(train_ar,test_ar)