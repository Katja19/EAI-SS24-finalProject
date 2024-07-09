import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow


@step(experiment_tracker="mlflow_experiment_tracker")
def evaluate_model(model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.DataFrame) -> Annotated[bool,"deployment_decision"]:
    """
    Evaluates the trained model and returns a deployment decision based on the out-of-sample accuracy.
    """
    
    # 1. Predict the test data
    y_pred = model.predict(X_test)
    
    # 2. Calculate the out-of-sample metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 3. Log the metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    # 4. Make a deployment decision based on the out-of-sample rmse
    if rmse < 10: # if the rmse is less than 10 persons of of the actual value, deploy the model
        deploy = True
    else:
        deploy = False
        
    return deploy