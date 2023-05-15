import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class FeatureEngineering:
    def __init__(self):
        self.threshold_high = 72 
        self.threshold_low = 50
    
    def get_new_features(self, df):
        """
        This function is resonsible for adding new features required for the model training.
        """
        try:
            # 1. Temperature range
            df['TemperatureRange'] = df['MaxOfUpperTRange'] - df['MinOfLowerTRange']    
            df['ExtremeHighTemp'] = (df['AverageOfUpperTRange'] > self.threshold_high).astype(int)
            df['ExtremeLowTemp'] = (df['AverageOfLowerTRange'] < self.threshold_low).astype(int)

            # 2. Total bee density
            df['TotalBeeDensity'] = df['honeybee'] + df['bumbles'] + df['andrena'] + df['osmia']

            # 3. Bee species dominance
            total_density = df['honeybee'] + df['bumbles'] + df['andrena'] + df['osmia']
            df['HoneybeeDominance'] = df['honeybee'] / total_density
            df['BumblesBeeDominance'] = df['bumbles'] / total_density
            df['AndrenaBeeDominance'] = df['andrena'] / total_density
            df['OsmiaBeeDominance'] = df['osmia'] / total_density

            # 4. Rain intensity
            df['RainIntensity'] = df['AverageRainingDays'] / df['RainingDays']

            # 5. Interaction features
            df['BeeDensity_TemperatureInteraction'] = df['TotalBeeDensity'] * df['TemperatureRange']
            df['BeeDensity_RainInteraction'] = df['TotalBeeDensity'] * df['RainIntensity']
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_feature_engineering(self, train_path, test_path):
        logging.info("Entered the data feature engineering method or component")
        try:
            # Read the dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read of train and test as dataframe completed")

            # Get new features as part of feature engineering
            train_df_fe = self.get_new_features(train_df)
            test_df_fe = self.get_new_features(test_df)
            logging.info("Feature Engineering of data is complete")

            return(
                train_df_fe,
                test_df_fe
            )
        except Exception as e:
            raise CustomException(e, sys)
