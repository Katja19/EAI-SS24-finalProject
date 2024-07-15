from zenml import step
from zenml.client import Client
from typing import Tuple
import pickle
import os
import wandb

@step
def get_model_and_preprocessing_pipeline(model_type:str) -> Tuple[object, object]:
    """
    Get the model and pipeline from the training pipeline and return it.
    """
    try:
        run = wandb.init(
            entity="ss24_eai",
            project="forecasting_model_multivariant",
        )
        client = Client()
        
        if model_type == 'xgboost':
            # model
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            # load pickle file
            with open(os.path.join(artifact_model_dir, 'm_xgboost_10_lags_10_trials_model.pkl'), 'rb') as f:
                model = pickle.load(f)
            
            # pipeline
            artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_pipeline:latest', type='pipeline')
            artifact_pipe_dir = artifact_pipe.download()

            pipeline = client.get_pipeline_from_dir(artifact_pipe_dir)
            
        elif model_type == 'random_forest':
            #model
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            # load pickle file
            with open(os.path.join(artifact_model_dir, 'm_random_forest_5_lags_50_trials_model.pkl'), 'rb') as f:
                model = pickle.load(f)
            
            # pipeline
            artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_pipeline:latest', type='pipeline')
            artifact_pipe_dir = artifact_pipe.download()

            pipeline = client.get_pipeline_from_dir(artifact_pipe_dir)
            
        else:
            raise ValueError(f'In the inference_pipeline the model_type {model_type} is not supported.')
        
    except Exception as e:
        print("Error in get_model_and_preprocessing_pipeline: ", e)
    
    
    return model, pipeline