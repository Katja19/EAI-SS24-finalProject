import pandas as pd
from zenml import step
import pickle
import os
import wandb
from zenml.client import Client

@step
def predictor(preprocessing_pipeline, 
              inference_data_original: pd.DataFrame, 
              model_type: str, 
              first_date_str: str, 
              last_date_str: str, 
              lags: int):
    """
    Predicts the next 24 hours of pedestrian counts using the trained model and the preprocessing pipeline.
    """
    try:
        run = wandb.init(
            entity="ss24_eai",
            project="forecasting_model_multivariant",
        )
        client = Client()
        
        # Load the appropriate model based on model_type
        if model_type == 'xgboost':
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            with open(os.path.join(artifact_model_dir, 'm_xgboost_10_lags_10_trials_model.pkl'), 'rb') as f:
                model = pickle.load(f)
        elif model_type == 'random_forest':
            artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_model:latest', type='model')
            artifact_model_dir = artifact_model.download()
            with open(os.path.join(artifact_model_dir, 'm_random_forest_5_lags_50_trials_model.pkl'), 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f'The model_type {model_type} is not supported.')

        print("MODEL_TYPE: ", type(model))

    except Exception as e:
        print("Error in get_model_and_preprocessing_pipeline: ", e)

    inference_data = inference_data_original.copy()
    
    predictions = []
    for i in range(24):
        print(f"\nIteration {i}")
        
        # Get the row of the inference data with the index i
        inference_data_row = inference_data.iloc[i]
        
        # Check for NaN values
        if inference_data_row.isnull().sum() > 0:
            print(f"Warning: There are {inference_data_row.isnull().sum()} NaN values in the inference data at index {i}.")
            continue
        
        # Preprocess the single row of the inference data
        try:
            preprocessed_data_row = preprocessing_pipeline.transform([inference_data_row])
        except Exception as e:
            print(f"Error during preprocessing at index {i}: {e}")
            continue
        
        # Make predictions on the preprocessed data
        try:
            predict_row = model.predict(preprocessed_data_row)[0]
            print(f"Predicted value at index {i}: {predict_row}")
        except Exception as e:
            print(f"Error during prediction at index {i}: {e}")
            continue

        # Update the predictions list
        predictions.append(predict_row)
        
        # Update the inference_data_original with the prediction
        inference_data_original.iloc[i, inference_data_original.columns.get_loc('pedestrians_count')] = predict_row
        print(f"Updated inference_data_original at index {i} with pedestrians_count: {predict_row}")
        
        # Update the lag features in inference_data
        if i + 1 < len(inference_data):
            inference_data.iloc[i + 1, inference_data.columns.get_loc('pedestrians_count_lag_1')] = predict_row
            print(f"Updated inference_data at index {i + 1} with pedestrians_count_lag_1: {predict_row}")
    
        for j in range(1, lags + 1):
            if i + 1 + j < len(inference_data):
                inference_data.iloc[i + 1 + j, inference_data.columns.get_loc(f'pedestrians_count_lag_{j + 1}')] = predict_row
                print(f"Updated inference_data at index {i + 1 + j} with pedestrians_count_lag_{j + 1}: {predict_row}")
    
    # Save the updated inference_data_original to a csv file
    inference_data_original.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

    
    # save the predictions to a csv file
    predictions_df = pd.DataFrame(predictions, columns=['pedestrians_count'])
    predictions_df.to_csv("predictions_values.csv", index=False)