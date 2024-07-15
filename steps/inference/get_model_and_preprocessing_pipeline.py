from zenml import step
from zenml.client import Client
from typing import Tuple
import pickle
import os
import wandb
import requests
from io import BytesIO

@step(experiment_tracker="wandb_experiment_tracker")
def get_model_and_preprocessing_pipeline(lags:int, trials:int, model_type:str) -> Tuple[object, object]:
    """
    Get the model and pipeline from the training pipeline and return it.
    """
    try:
        #run = wandb.init(
        #    entity="ss24_eai",
        #    project="forecasting_model_multivariant"
        #)
        run_name = f"m_{model_type}_{lags}_lags_{trials}_trials"
        wandb.init(project="forecasting_model_multivariant", name=run_name)
        
        client = Client()
        
        if model_type == 'xgboost':
            # model
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            print(os.listdir(artifact_model_dir))
            # load pickle file
            path = os.path.join(artifact_model_dir, 'm_xgboost_10_lags_10_trials_model.pkl')
            with open(path, 'rb') as f:
                model = pickle.load(f)
                if model is not None:
                    print("Model loaded successfully.")
                else:
                    print("Model is None.")
            
            # pipeline
            #artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_pipeline:latest', type='pipeline')
            #pipeline_url = artifact_pipe.direct_url
            
        elif model_type == 'random_forest':
            #model
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            print(os.listdir(artifact_model_dir))
            # load pickle file
            path = os.path.join(artifact_model_dir, 'm_random_forest_5_lags_50_trials_model.pkl')
            print(path)
            with open(path, 'rb') as f:
                model = pickle.load(f)
                if model is not None:
                    print("Model loaded successfully.")
                else:
                    print("Model is None.")
            
            # pipeline
            #artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_pipeline:latest', type='pipeline')
            #pipeline_url = artifact_pipe.direct_url
            
        else:
            raise ValueError(f'In the inference_pipeline the model_type {model_type} is not supported.')
        
        # Download pipeline from WandB artifact URL
        #response = requests.get(pipeline_url)
        #if response.status_code == 200:
        #    pipeline = pickle.load(BytesIO(response.content))
        #else:
        #    raise Exception(f"Error downloading pipeline from WandB artifact URL: {pipeline_url}")
        
        preprocessing_pipeline = client.get_artifact_version("pipeline")
        
    except Exception as e:
        print("Error in get_model_and_preprocessing_pipeline: ", e)
    
    
    return model, preprocessing_pipeline