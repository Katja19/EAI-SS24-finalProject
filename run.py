from pipelines.training_pipeline import training_pipeline
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline
#from pipelines.inference_pipeline import inference_pipeline

def run_pipelines(model_variant, model_type):
    
    # Execute the feature engineering pipeline
    # Result: 
    # X_train, X_test, y_train, y_test, input_data, 
    # X_train_preprocessed, X_test_preprocessed, y_train,n y_test, pipeline are saved as artifacts
    feature_engineering_pipeline()
    
    # Execute the training pipeline
    # Result: 
    # deoployed model, best hyperparameters, in-sample RMSE and deployment decision are saved as artifacts
    training_pipeline(model_variant=model_variant, model_type=model_type)
    
    # Execute the inference pipeline
    #inference_pipeline(model_variant=model_variant, model_type=model_type)
    #inference_pipeline()
