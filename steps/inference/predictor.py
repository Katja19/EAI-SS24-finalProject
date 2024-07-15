import pandas as pd
from zenml import step

@step
def predictor(model, preprocessing_pipeline, inference_data_original:pd.DataFrame, model_type:str, first_data_str:str, last_data_str:str, lags:int):
    """
    Predicts the next 24 hours of pedestrian counts using the trained model and the preprocessing pipeline.
    """
    print("Starting predictor step...")
    print("model_type: ", type(model))
    print("preprocessing_pipeline: ", type(preprocessing_pipeline))
    # 7. recursive preprocessing, forcasting an updating the inference data
    
    #drop the columns that we dont want to be preprocess and the target variable = 'pedestrains_count'!
    # add copy of loaction_id to the inference data, cause we need it for the recursive forecasting
    inference_data_original['location_id_copy'] = inference_data_original['location_id']
    inference_data = inference_data_original.drop(columns=['location_id_copy', 'timestamp', 'date', 'datetime', 'pedestrains_count'])
    
    predictions = []
    for i in range(24):
        # 7.1 preprocess the inference data so that it can be used as input for the model
        # get the rows of the inference data with the index i
        inference_data_row = inference_data.iloc[i]
        # check for nan values
        if inference_data_row.isnull().sum() > 0:
            print(f"Warning: There are {inference_data_row.isnull().sum()} nan values in the inference data.")
        else:
            # preprocess the the singel row of the inference data
            preprocessed_data_row = preprocessing_pipeline.transform(inference_data_row)
            
        # 7.2 make predictions on the preprocessed data
        predict_row = model.predict(preprocessed_data_row)[0]

        # 7.3 update the inference data with the predictions
        predictions.append(predict_row)
        
        # update the inference data original with the predictions, this has no lag features
        inference_data_original['pedestrians_count'].iloc[i] = predict_row
        inference_data.at[i+1, 'pedestrians_count_lag_1'] = predict_row
    
        # update the lag features of the inference data
        for j in range(1, lags+1):
            if i + 1 + j < 24:
                inference_data.at[i+1+j, f'pedestrians_count_lag_{j+1}'] = predict_row
            
                
        
        #for lag in range(1, lags+1):
        #    #inference_data_original['pedestrians_count_lag_'+str(lag)].iloc[i] = inference_data_original['pedestrians_count'].shift(lag)
        #    inference_data['pedestrians_count_lag_'+str(lag)].iloc[i] = inference_data['pedestrians_count'].shift(lag)
            
    # 8. save the inference data with the predictions as an csv file
    inference_data_original.to_csv(f"data/inference_{model_type}_{first_data_str}_to_{last_data_str}_with_predictions.csv", index=False)