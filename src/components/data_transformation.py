import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class Datatransformationconfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

class Datatransformation:
    def __init__(self):
        self.data_transformations_config=Datatransformationconfig()
        

    def get_data_transformer_obj(self):
        try:
            num_features=['age', 'bmi', 'children']

            cat_features=['sex', 'smoker', 'region']


            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                    
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical column encoding done")
            logging.info("Numerical colmn scaling completed")


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_features',cat_pipeline,cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = 'charges'

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df = test_df[target_column_name]

            input_features_train_array=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_array = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_array,np.array(target_features_train_df)
            ]
            test_arr = np.c_[
                input_features_test_array,np.array(target_features_test_df)
            ]

            logging.info('Saved preprocessing object')

            save_object(

                file_path = self.data_transformations_config.preprocessor_file_path,
                obj = preprocessor_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformations_config.preprocessor_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
