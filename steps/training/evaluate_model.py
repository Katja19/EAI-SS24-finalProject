import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#import mlflow
import wandb
import logging

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step(experiment_tracker="wandb_experiment_tracker")
def evaluate_model(model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.DataFrame, model_variant:str, model_type:str, trials:int, 
                   in_sample_rmse:float, best_parameters:dict) -> Annotated[bool, "Deployment Decision"]:
    """
    Evaluates the trained model and returns a deployment decision based on the out-of-sample accuracy.
    """
    
    logger.info("Starting evaluate_model step...")
    
    # 1. Predict the test data
    y_pred = model.predict(X_test)
    
    # 2. Calculate the out-of-sample metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # OLD3. Log the metrics
    # mlflow.log_metric("mse", mse)
    # mlflow.log_metric("rmse", rmse)
    # mlflow.log_metric("r2", r2)
    # mlflow.log_metric("mae", mae)
    
    # 3. Log the metrics to wandb
    #wandb.log({"mse": mse, "rmse": rmse, "r2": r2, "mae": mae})
    #logger.info(f"Metrics logged to wandb. RMSE: {rmse}")
    
    # 4. Make a deployment decision based on the out-of-sample rmse
    # get the max_prededestrians value from the y_test data   
    #TODO: get the max value from the y_test data
    
    
    if rmse < 10: # if the rmse is less than 10 persons of of the actual value, deploy the model
        deploy = True
    else:
        deploy = False
        
    # 3. Log everything of the run to wandb
    wandb.init(project="forcasting_model_multivariant", name=f"{model_variant}_{model_type}_{trials}_trials")
    wandb.log({"model_variant": model_variant,
                "model_type": model_type,
                "trials": trials, 
                "in_sample_rmse": in_sample_rmse,
                "rmse": rmse,
                "mse": mse,
                "r2": r2,
                "mae": mae,
                "deployment_decision": deploy})
    
    for key, value in best_parameters.items():
        wandb.log({key: value})
        
    if deploy:
        wandb.log({"model": model})
        
    logger.info(f"Deployment decision: {deploy}")
    logger.info("Finished evaluate_model step.")
    
    wandb.finish()
        
    return deploy