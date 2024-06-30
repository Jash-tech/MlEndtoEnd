import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import load_object


class PredictPrediction():
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            logging.info("Path Created for Model and preprocessor Obj")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            logging.info("Loading Model and preprocessing Object")

            sca_features=preprocessor.transform(features)
            pred=model.predict(sca_features)
            logging.info("Features Transformed and predicted")

            return pred

        except Exception as e:
            raise CustomException(e,sys)

class CustomData():

    def __init__(self, ClaimNb: int, Exposure: float, Power: str, CarAge: int, DriverAge: int, 
                Brand: str, Gas: str, Region: str, Density: int, ClaimFreq: float):
        self.ClaimNb = ClaimNb
        self.Exposure = Exposure
        self.Power = Power
        self.CarAge = CarAge
        self.DriverAge = DriverAge
        self.Brand = Brand
        self.Gas = Gas
        self.Region = Region
        self.Density = Density
        self.ClaimFreq = ClaimFreq

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            'ClaimNb': [self.ClaimNb],
            'Exposure': [self.Exposure],
            'Power': [self.Power],
            'CarAge': [self.CarAge],
            'DriverAge': [self.DriverAge],
            'Brand': [self.Brand],
            'Gas': [self.Gas],
            'Region': [self.Region],
            'Density': [self.Density],
            'ClaimFreq': [self.ClaimFreq]
             }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')

            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

        



