from zenml import pipeline
from steps import update_data
from steps import load_data_inference, create_derived_features_inference, create_inference_data, get_model_and_preprocessing_pipeline, predictor

@pipeline(enable_cache=False)
def inference_pipeline(model_type:str, lags:int):
    """
    Runs the inference pipeline to predict the target variable of the inference data
    """
    
    if model_type == 'xgboost':
        lags = 10
    elif model_type == 'random_forest':
        lags = 5
    
    # 0. ensue the execution order of the steps
    #load_data_inference.after(update_data)
    create_derived_features_inference.after(load_data_inference)
    create_inference_data.after(create_derived_features_inference)
    get_model_and_preprocessing_pipeline.after(create_inference_data)
    predictor.after(get_model_and_preprocessing_pipeline)

    # 1. we update the data in the database using the update_data step from the feature_engineering pipeline
    #update_data()
    
    # 2. we load the data similar to the load_data step from the feature_engineering pipeline,
    # but this time we dont drop rows with null values, cause we need them for the recursive forecasting
    dataset_hist, event_dataset = load_data_inference()
    
    # 3. we create derived features for the inference data using the create_inference_data step
    dataset_hist, event_dataset  = create_derived_features_inference(dataset_hist, event_dataset, lags) 
    
    # 4. we create the inference data for the next 24 hours that we want to predict 
    # Hint: there are nan values in it cause they will be filled during the recursive forecasting with the model
    inference_data_original, first_date_str, last_date_str = create_inference_data(dataset_hist, event_dataset, lags, model_type)
    
    # 5. we save the inference data as an csv file
    #inference_data_original.to_csv(f"data/inference_{model_type}_{first_data_str}_to_{last_data_str}.csv", index=False)
    
    # 6. get the model and the preprocessing pipeline from wandb
    model, preprocessing_pipeline = get_model_and_preprocessing_pipeline(model_type)
    
    # 7. prediction: recursive preprocessing, forcasting an updating the inference data
    predictor(model, preprocessing_pipeline, inference_data_original, model_type, first_date_str, last_date_str, lags)
    
    # 8. plot the predictions using the csv file with the predictions
    # no, we will not implement this step in the main notebook, cause it is not necessary for the main goal of the project