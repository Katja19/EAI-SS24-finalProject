from pipelines.training_pipeline import training_pipeline
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline
#from pipelines.inference_pipeline import inference_pipeline
import logging
import wandb

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipelines(model_variant, model_type):
    
    # Execute the feature engineering pipeline
    # Result: 
    # X_train, X_test, y_train, y_test, input_data, 
    # X_train_preprocessed, X_test_preprocessed, y_train,n y_test, pipeline are saved as artifacts
    logger.info("Starting the feature engineering pipeline.")
    feature_engineering_pipeline()
    logger.info("Feature engineering pipeline completed.")
    
    # Execute the training pipeline
    # Result: 
    # deoployed model, best hyperparameters, in-sample RMSE and deployment decision are saved as artifacts
    logger.info("Starting the training pipeline.")
    
    # Initialize wandb run for logging the training pipeline results, it is initialized here because 
    # we want to log the results of the training pipeline to a new run but in the same project
    #wandb.init(project="forcasting_model_multivariant", name="default_run_name")

    
    training_pipeline(model_variant=model_variant, model_type=model_type)
    logger.info("Training pipeline completed.")
    
    # Execute the inference pipeline # wird in der main seperat später ausgeführt
    #inference_pipeline(model_variant=model_variant, model_type=model_type)
    #inference_pipeline()
