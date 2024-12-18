from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model
from src.utils import save_object

@dataclass
class Modeltrainerconfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class Model_trainer():
    def __init__(self):
        self.model_config = Modeltrainerconfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                         train_array[:, :-1],  # All columns except the last as features
                        train_array[:, -1],   # Last column as target
                      test_array[:, :-1],   # Same for test data
                        test_array[:, -1]
            )



            models = {
                "LineraReagression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "SVR":SVR(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "XGBRegressor":XGBRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info("Model training completed and best model found")

            save_object(file_path=self.model_config.trained_model_path,
                        obj=best_model)
            Predicted = best_model.predict(X_test)

            final_r2_score = r2_score(y_test,Predicted)

            return final_r2_score


        except Exception as e:
            raise CustomException(e,sys)
            