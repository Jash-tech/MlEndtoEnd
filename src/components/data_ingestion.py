import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            freq=pd.read_csv(r'C:\Users\LENOVO\Downloads\archive_i\freMTPLfreq.csv')
            sev=pd.read_csv(r'C:\Users\LENOVO\Downloads\archive_i\freMTPLsev.csv')
            logging.info('Reading two files')

            MTPLsev_grp = sev.groupby(['PolicyID'])[['ClaimAmount']].agg('sum').reset_index()
            df_merged = pd.merge(freq, MTPLsev_grp, how='outer', on='PolicyID').fillna(0).set_index('PolicyID')
            df_merged['ClaimFreq'] = df_merged['ClaimNb'] / df_merged['Exposure']
            data=df_merged.copy()
            logging.info('Basic Preprocessing data')

            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)),exist_ok=True)
            logging.info("Creating Directory")

            data.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw dataset in artifact folder")

            train_data,test_data=train_test_split(data,test_size=0.25,random_state=42)
            logging.info("Splitting the data in Train and Test")

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info("Saved the Train and Test dataset in artifact folder")

            logging.info("Data Ingestion Completed")

            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path)




        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=='__main__':
#     a=DataIngestion()
#     a.initiate_data_ingestion()
