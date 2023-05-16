import sys
import pandas as pd
from src.components.feature_engineering import FeatureEngineering
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            feature_selector_path = 'artifacts\selector.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            feature_selector = load_object(file_path = feature_selector_path)
            feature_engineering = FeatureEngineering()
            df_fe = feature_engineering.get_new_features(features)
            data_scaled = preprocessor.transform(df_fe)
            data_scaled_fe = feature_selector.transform(data_scaled)
            preds = model.predict(data_scaled_fe)
            return preds
        except Exception as e:
            raise CustomException(e, sys)        


class CustomData:
    def __init__(self,
                 clone_size: float,
                 honeybee: float,
                 bumbles: float,
                 andrena: float,
                 osmia: float,
                 Max_Upper_TRange: float,
                 Min_Upper_TRange: float,
                 Average_Upper_TRange: float,
                 Max_Lower_TRange: float,
                 Min_Lower_TRange: float,
                 Average_Lower_TRange: float,
                 Raining_Days: float,
                 Average_Raining_Days: float,
                 fruit_set: float,
                 fruit_mass: float,
                 seeds: float):
        self.clone_size = clone_size
        self.honeybee = honeybee
        self.bumbles = bumbles
        self.andrena = andrena
        self.osmia = osmia
        self.Max_Upper_TRange = Max_Upper_TRange
        self.Min_Upper_TRange = Min_Upper_TRange
        self.Average_Upper_TRange = Average_Upper_TRange
        self.Max_Lower_TRange = Max_Lower_TRange
        self.Min_Lower_TRange = Min_Lower_TRange
        self.Average_Lower_TRange = Average_Lower_TRange
        self.Raining_Days = Raining_Days
        self.Average_Raining_Days = Average_Raining_Days
        self.fruit_set = fruit_set
        self.fruit_mass = fruit_mass
        self.seeds = seeds

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "clonesize": [self.clone_size],
                "honeybee": [self.honeybee],
                "bumbles": [self.bumbles],
                "andrena": [self.andrena],
                "osmia": [self.osmia],
                "MaxOfUpperTRange": [self.Max_Upper_TRange],
                "MinOfUpperTRange": [self.Min_Upper_TRange],
                "AverageOfUpperTRange": [self.Average_Upper_TRange],
                "MaxOfLowerTRange": [self.Max_Lower_TRange],
                "MinOfLowerTRange": [self.Min_Lower_TRange],
                "AverageOfLowerTRange": [self.Average_Lower_TRange],
                "RainingDays": [self.Raining_Days],
                "AverageRainingDays": [self.Average_Raining_Days],
                "fruitset": [self.fruit_set],
                "fruitmass": [self.fruit_mass],
                "seeds": [self.seeds],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)