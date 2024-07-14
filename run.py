from pipelines.training_pipeline import training_pipeline
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline
#from pipelines.inference_pipeline import inference_pipeline
import logging
import wandb

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipelines(model_variant, model_type, lags, trials):
    
    logger.info("Starting the feature engineering pipeline.")
    feature_engineering_pipeline(model_variant=model_variant, model_type=model_type, lags=lags, trials=trials)
    logger.info("Feature engineering pipeline completed.")
    

    logger.info("Starting the training pipeline.")
    training_pipeline(model_variant=model_variant, model_type=model_type,lags=lags, trials=trials)
    logger.info("Training pipeline completed.")
    
    
    # Execute the inference pipeline # wird in der main seperat später ausgeführt
    #inference_pipeline(model_variant=model_variant, model_type=model_type)
    #inference_pipeline()
    
#def run_inference_pipeline(lags):
#    # Execute the inference pipeline
#    inference_pipeline(lags)
