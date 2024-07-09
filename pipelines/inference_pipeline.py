from zenml import pipeline
from zenml.client import Client
from steps import prediction_service_loader, predictor, inference_data_loader, inference_preprocessing

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Runs the inference pipeline to predict the target variable of the inference data
    """
    
    # 1. Load the inference data, these are the data points for which we want to predict the target variable
    # In our case: The next 24 hours of the time series = the next day
    data = inference_data_loader("./data/inference.csv")
    
    # 2. Get the preprocessing pipeline from the training pipeline
    client = Client()
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    
    # 3. Preprocess the inference data so that it can be used as input for the model
    preprocessed_data = inference_preprocessing(preprocessing_pipeline,data)
    
    # 4. Load the model deployment service, from this we get the deployed model we want to use for inference
    model_deployment_service = prediction_service_loader(
        pipeline_name="training_pipeline",
        step_name="mlflow_model_deployer_step",
    )
    
    # 5. Use the model deployment service to make predictions on the preprocessed data, here we save the predictions as an artifact
    prediction = predictor(service=model_deployment_service, input_data=preprocessed_data)