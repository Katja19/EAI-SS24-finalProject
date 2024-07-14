import pandas as pd
#import mlflow
import wandb
from zenml import step
from typing_extensions import Annotated
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Tuple
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
import logging

@step#(experiment_tracker="wandb_experiment_tracker") # getting this later in inference_pipeline.py
def model_trainer(X_train: pd.DataFrame, y_train: pd.Series, model_type: str, best_parameters: dict) -> Tuple[Annotated[RegressorMixin, "Model"], Annotated[float, "In_Sample_RMSE"]]:
    """
    Trains a regression model using the training dataset and the best hyperparameters found during hyperparameter tuning.
    """
    # OLD:1. First, we autolog the model training process using MLflow
    #mlflow.sklearn.autolog()
    
    # 1. Initialize wandb run for logging
    #wandb.init(project="eai_project", name="model_training")

    # 2. Choose the model class based on the model_type
    if model_type == 'random_forest':
        model = RandomForestRegressor(**best_parameters)
    elif model_type == 'xgboost':
        model = XGBRegressor(**best_parameters)
    else:
        raise ValueError("Unsupported model_type. Supported types: 'random_forest', 'xgboost'")

    # 3. Train the model
    model.fit(X_train, y_train)

    # 4. Calculate the in-sample rmse
    y_pred = model.predict(X_train)
    in_sample_rmse = mean_squared_error(y_train, y_pred, squared=False) # squared=False returns the RMSE
    
    # 5. Log the in-sample rmse to wandb
    #wandb.log({"in_sample_rmse": in_sample_rmse})
    
    # 6. Finish the wandb run
    #wandb.finish()

    return model, in_sample_rmse
