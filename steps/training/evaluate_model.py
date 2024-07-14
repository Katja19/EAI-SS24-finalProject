import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from zenml.client import Client

#import mlflow
import wandb
import logging

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step(experiment_tracker="wandb_experiment_tracker")
def evaluate_model(model:RegressorMixin,
                   X_test:pd.DataFrame,
                   y_test:pd.DataFrame, 
                   model_variant:str, 
                   model_type:str, 
                   trials:int, 
                   in_sample_rmse:float,
                   lags:int,
                   best_parameters:dict) -> Annotated[bool, "Deployment Decision"]:
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

    
    # 3. Make a deployment decision based on out-of-sample RMSE
    # depends on your business case, so we will not set a fixed threshold here    
    deploy = True
    
    # get first char of model_variant
    model_variant_char = model_variant[0]
        
    # 4. Log everything of the run to wandb
    wandb.init(project="forecasting_model_multivariant", name=f"{model_variant_char}_{model_type}_{lags}_lags_{trials}_trials")
    wandb.log({"model_variant": model_variant,
                "model_type": model_type,
                "trials": trials, 
                "in_sample_rmse": in_sample_rmse,
                "rmse": rmse,
                "mse": mse,
                "r2": r2,
                "mae": mae,
                "lags":lags,
                "deployment_decision": deploy})
    
    for key, value in best_parameters.items():
        wandb.log({key: value})
        
    
    # 5. Save the model if deployment decision is True    
    if deploy:
        
        # Delete previous model.pkl if it exists, cause else the new model will be added to the old model
        if os.path.exists("model.pkl"):
            os.remove("model.pkl")
        
        
        #open a file and save the model, if file doest exist it is created automatically
        with open(f"model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        model_artifact = wandb.Artifact(f"{model_variant}_{model_type}_{trials}_trials", type="model") # create a model artifact
        model_artifact.add_file("model.pkl")
        wandb.log_artifact(model_artifact) # this will save the model artifact to wandb
        
    # 6. save the fitted pipeline in wandb
    # get the pipeline from zenml
    client = Client()
    pipeline_version = client.get_artifact_version("pipeline")
    pipeline_dir = pipeline_version.uri
    print("pipeline_version: ", pipeline_version)
    print(type(pipeline_version))
    
    # save the pipeline to wandb
    pipeline_artifact = wandb.Artifact(f"{model_variant}_{model_type}_{trials}_trials", type="pipeline")
    pipeline_artifact.add_dir(pipeline_dir)
    wandb.log_artifact(pipeline_artifact)
    
        
    logger.info(f"Deployment decision: {deploy}")
    logger.info("Finished evaluate_model step.")
    
    wandb.finish()
        
    return deploy