from steps import hp_tuning,model_trainer,evaluate_model
from zenml import pipeline
from zenml.client import Client
#from zenml.integrations.mlflow.steps import mlflow_model_deployer_step # da on windows not working
#from zenml.materializers.pandas_materializer import PandasMaterializer
import wandb
import logging
import os

""" 
This pipeline uses the data from the Feature Engineering pipeline. 
We can get this data using our ZenML client, as we did before, to extract the mae value.
It then creates a decision tree and deploys it using MLflow when a deployment decision is made.
"""

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pipeline#(experiment_tracker="wandb_experiment_tracker")
def training_pipeline(model_variant:str, model_type:str, lags:int, trials:int):
    """ 
        Pipeline to train and deploy a machine learning model using preprocessed and encoded datasets.
    """
    
    logger.info("Starting training_pipeline...")
    
    try:
    
        # 1. Get the preprocessed X data and the non-encoded y data from the Feature Engineering pipeline
        client = Client()
        X_train = client.get_artifact_version("X_train_preprocessed")
        X_test = client.get_artifact_version("X_test_preprocessed")
        y_train = client.get_artifact_version("y_train") # not encoded cause it is a regression problem
        y_test = client.get_artifact_version("y_test") # not encoded cause it is a regression problem
        
        # 2. Get the best hyperparameters for the model
        best_parameters = hp_tuning(X_train,y_train,model_type, trials=trials)
        
        # 3. Train the model and get the in-sample score (RMSE)
        model,in_sample_rmse =model_trainer(X_train,y_train,model_type,best_parameters)
        
        # 4. Evaluate the model using the test data, here we calculate and save the out-of-sample score (MSE) and other metrics
        #deploy, rmse, mse, r2, mae = evaluate_model(model,X_test,y_test)
        #deploy = evaluate_model(model,X_test,y_test, model_variant, model_type, trials, in_sample_rmse, lags, trials, best_parameters)
        deploy = evaluate_model(model=model, X_test=X_test, y_test=y_test, model_variant=model_variant, model_type=model_type, trials=trials, in_sample_rmse=in_sample_rmse, lags=lags, best_parameters=best_parameters)
            
        logger.info("Finished training_pipeline.")
            
    except Exception as e:
        logger.error("Error in training_pipeline: ", e)