# from pipelines.training_pipeline import training_pipeline
# from pipelines.feature_engineering_pipeline import feature_engineering_pipeline
# from pipelines.inference_pipeline import inference_pipeline

# if __name__ == "__main__":
    
#     # this pipeline saves the raw data (X_train, X_test, y_train, y_test, input_data)
#     # but also the preprocessed data (X_train_preprocessed, X_test_preprocessed, y_train_encoded, y_test_encoded)
#     # but also the prepro_pipeline, which is the pipeline that was used to preprocess the data
#     feature_engineering_pipeline()
    
#     # TODO
#     training_pipeline()
    
#     # TODO
#     inference_pipeline()
      
import argparse
from pipelines.training_pipeline import training_pipeline
from pipelines.feature_engineering_pipeline import feature_engineering_pipeline
from pipelines.inference_pipeline import inference_pipeline

def main(model_variant, model_type):
    
    # Execute the feature engineering pipeline
    # Result: 
    # X_train, X_test, y_train, y_test, input_data, 
    # X_train_preprocessed, X_test_preprocessed, y_train,n y_test, pipeline are saved as artifacts
    #feature_engineering_pipeline(model_variant=model_variant, model_type=model_type)
    feature_engineering_pipeline()
    
    # Execute the training pipeline
    # Result: 
    # deoployed model, best hyperparameters, in-sample RMSE and deployment decision are saved as artifacts
    training_pipeline(model_variant=model_variant, model_type=model_type)
    
    # Execute the inference pipeline
    #inference_pipeline(model_variant=model_variant, model_type=model_type)
    inference_pipeline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute pipelines with model parameters')
    parser.add_argument('--model_variant', type=str, choices=['multi', 'uni'], default='multi',
                        help='Model variant ("multi" or "uni")')
    parser.add_argument('--model_type', type=str, choices=['xgb', 'arima'], default='xgb',
                        help='Model type ("xgb" or "arima")')
    
    args = parser.parse_args()
    main(args.model_variant, args.model_type)