# import pandas as pd
# from zenml import step

# @step
# def predictor(model, 
#               preprocessing_pipeline, 
#               inference_data_original:pd.DataFrame, 
#               model_type:str, 
#               first_data_str:str, 
#               last_data_str:str, 
#               lags:int):
#     """
#     Predicts the next 24 hours of pedestrian counts using the trained model and the preprocessing pipeline.
#     """
#     print("Starting predictor step...")
#     print("model_type: ", type(model))
#     print("preprocessing_pipeline: ", type(preprocessing_pipeline))
#     # 7. recursive preprocessing, forcasting an updating the inference data
    
#     #drop the columns that we dont want to be preprocess and the target variable = 'pedestrains_count'!
#     # add copy of loaction_id to the inference data, cause we need it for the recursive forecasting
#     inference_data_original['location_id_copy'] = inference_data_original['location_id']
#     inference_data = inference_data_original.drop(columns=['location_id_copy', 'timestamp', 'date', 'datetime', 'pedestrains_count'])
    
#     predictions = []
#     for i in range(24):
#         # 7.1 preprocess the inference data so that it can be used as input for the model
#         # get the rows of the inference data with the index i
#         inference_data_row = inference_data.iloc[i]
#         # check for nan values
#         if inference_data_row.isnull().sum() > 0:
#             print(f"Warning: There are {inference_data_row.isnull().sum()} nan values in the inference data.")
#         else:
#             # preprocess the the singel row of the inference data
#             preprocessed_data_row = preprocessing_pipeline.transform(inference_data_row)
            
#         # 7.2 make predictions on the preprocessed data
#         predict_row = model.predict(preprocessed_data_row)[0]

#         # 7.3 update the inference data with the predictions
#         predictions.append(predict_row)
        
#         # update the inference data original with the predictions, this has no lag features
#         inference_data_original['pedestrians_count'].iloc[i] = predict_row
#         inference_data.at[i+1, 'pedestrians_count_lag_1'] = predict_row
    
#         # update the lag features of the inference data
#         for j in range(1, lags+1):
#             if i + 1 + j < 24:
#                 inference_data.at[i+1+j, f'pedestrians_count_lag_{j+1}'] = predict_row
            
                
        
#         #for lag in range(1, lags+1):
#         #    #inference_data_original['pedestrians_count_lag_'+str(lag)].iloc[i] = inference_data_original['pedestrians_count'].shift(lag)
#         #    inference_data['pedestrians_count_lag_'+str(lag)].iloc[i] = inference_data['pedestrians_count'].shift(lag)
            
#     # 8. save the inference data with the predictions as an csv file
#     inference_data_original.to_csv(f"data/inference_{model_type}_{first_data_str}_to_{last_data_str}_with_predictions.csv", index=False)
    
    ############################################################################################################
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
        return

    print("Starting predictor step...")
    print("preprocessing_pipeline: ", type(preprocessing_pipeline))
    print("inference_data_original: ", type(inference_data_original))
    print("lags: ", type(lags))

    # Prepare the inference data by dropping unnecessary columns
    inference_data_original['location_id_copy'] = inference_data_original['location_id']
    inference_data = inference_data_original.drop(columns=['location_id_copy', 'timestamp', 'date', 'datetime', 'pedestrians_count'])
    
    predictions = []
    for i in range(24):
        # Get the rows of the inference data with the index i
        inference_data_row = inference_data.iloc[i]
        
        # Check for NaN values
        if inference_data_row.isnull().sum() > 0:
            print(f"Warning: There are {inference_data_row.isnull().sum()} NaN values in the inference data.")
            continue
        
        # Preprocess the single row of the inference data
        try:
            preprocessed_data_row = preprocessing_pipeline.transform([inference_data_row])
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            continue
        
        # Make predictions on the preprocessed data
        try:
            predict_row = model.predict(preprocessed_data_row)[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        # Update the inference data with the predictions
        predictions.append(predict_row)
        
        # Update the inference data original with the predictions, this has no lag features
        inference_data_original.loc[i, 'pedestrians_count'] = predict_row
        if i + 1 < len(inference_data):
            inference_data.at[i + 1, 'pedestrians_count_lag_1'] = predict_row
    
        # Update the lag features of the inference data
        for j in range(1, lags + 1):
            if i + 1 + j < len(inference_data):
                inference_data.at[i + 1 + j, f'pedestrians_count_lag_{j + 1}'] = predict_row
    
    # Save the inference data with the predictions as a CSV file
    first_date_day = first_date_str.split('T')[0]
    first_date_time = first_date_str.split('T')[1]
    last_date_day = last_date_str.split('T')[0]
    last_date_time = last_date_str.split('T')[1]
    
    # output_path = f"data/inference_{model_type}_{first_date_day}_{first_date_time}_to_{last_date_day}_{last_date_time}_with_predictions.csv"
    # # change alls : to - in the output_path
    # output_path = output_path.replace(':', '-')
    # inference_data_original.to_csv(output_path, index=False)
    # print(f"Predictions saved to {output_path}")
    
    # save dataframe inference_data_original to a csv file
    inference_data_original.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")




# ######################################################
# import pandas as pd
# from zenml import step
# import pickle
# import os
# import wandb
# from zenml.client import Client

# @step
# def predictor(#model: object, 
#               preprocessing_pipeline, 
#               inference_data_original: pd.DataFrame, 
#               model_type: str, 
#               first_date_str: str, 
#               last_date_str: str, 
#               lags: int):
#     """
#     Predicts the next 24 hours of pedestrian counts using the trained model and the preprocessing pipeline.
#     """
#     ############
#     try:
#         run = wandb.init(
#             entity="ss24_eai",
#             project="forecasting_model_multivariant",
#         )
#         client = Client()
        
#         if model_type == 'xgboost':
#             # model
#             artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_model:latest', type='model')
#             artifact_model_dir = artifact_model.download()
#             # load pickle file
#             with open(os.path.join(artifact_model_dir, 'm_xgboost_10_lags_10_trials_model.pkl'), 'rb') as f:
#                 model = pickle.load(f)
            
#             # pipeline
#             #artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_xgboost_10_lags_10_trials_pipeline:latest', type='pipeline')
#             #pipeline_url = artifact_pipe.direct_url
            
#         elif model_type == 'random_forest':
#             #model
#             artifact_model = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_model:latest', type='model')
#             artifact_model_dir = artifact_model.download()
#             # load pickle file
#             with open(os.path.join(artifact_model_dir, 'm_random_forest_5_lags_50_trials_model.pkl'), 'rb') as f:
#                 model = pickle.load(f)
            
#             # pipeline
#             #artifact_pipe = run.use_artifact('ss24_eai/forecasting_model_multivariant/m_random_forest_5_lags_50_trials_pipeline:latest', type='pipeline')
#             #pipeline_url = artifact_pipe.direct_url
            
#         else:
#             raise ValueError(f'In the inference_pipeline the model_type {model_type} is not supported.')
        
#         # # Download pipeline from WandB artifact URL
#         # response = requests.get(pipeline_url)
#         # if response.status_code == 200:
#         #     pipeline = pickle.load(BytesIO(response.content))
#         # else:
#         #     raise Exception(f"Error downloading pipeline from WandB artifact URL: {pipeline_url}")
        
#         print("MODEL_TYPE: ", type(model))
        
#     except Exception as e:
#         print("Error in get_model_and_preprocessing_pipeline: ", e)
#     #########
#     print("Starting predictor step...")
#     #print("model_type: ", type(model))
#     print("preprocessing_pipeline: ", type(preprocessing_pipeline))
#     print("inference_data_original: ", type(inference_data_original))
#     print("lags: ", type(lags))

#     # 7. recursive preprocessing, forecasting and updating the inference data
    
#     #drop the columns that we dont want to be preprocess and the target variable = 'pedestrians_count'!
#     # add copy of location_id to the inference data, cause we need it for the recursive forecasting
#     inference_data_original['location_id_copy'] = inference_data_original['location_id']
#     inference_data = inference_data_original.drop(columns=['location_id_copy', 'timestamp', 'date', 'datetime', 'pedestrians_count'])
    
#     predictions = []
#     for i in range(24):
#         # 7.1 preprocess the inference data so that it can be used as input for the model
#         # get the rows of the inference data with the index i
#         inference_data_row = inference_data.iloc[i]
#         # check for nan values
#         if inference_data_row.isnull().sum() > 0:
#             print(f"Warning: There are {inference_data_row.isnull().sum()} nan values in the inference data.")
#         else:
#             # preprocess the single row of the inference data
#             preprocessed_data_row = preprocessing_pipeline.transform(inference_data_row)
            
#         # 7.2 make predictions on the preprocessed data
#         predict_row = model.predict(preprocessed_data_row)
#         #predict_row = 0

#         # 7.3 update the inference data with the predictions
#         predictions.append(predict_row)
        
#         # update the inference data original with the predictions, this has no lag features
#         inference_data_original['pedestrians_count'].iloc[i] = predict_row
#         inference_data.at[i+1, 'pedestrians_count_lag_1'] = predict_row
    
#         # update the lag features of the inference data
#         for j in range(1, lags+1):
#             if i + 1 + j < 24:
#                 inference_data.at[i+1+j, f'pedestrians_count_lag_{j+1}'] = predict_row
    
#     # 8. save the inference data with the predictions as a csv file
#     first_date_day = first_date_str.split('T')[0]
#     first_date_time = first_date_str.split('T')[1]
#     last_date_day = last_date_str.split('T')[0]
#     last_date_time = last_date_str.split('T')[1]
    
#     inference_data_original.to_csv(f"data/inference_{model_type}_{first_date_day}_{first_date_time}_to_{last_date_day}_{last_date_time}_with_predictions.csv", index=False)
    