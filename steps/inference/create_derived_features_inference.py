from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def create_derived_features_inference(dataset:pd.DataFrame, event_dataset:pd.DataFrame, lags:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create derived features from the dataset.
    Here we create lag features from the target column and
    time-based features.
    """
    
    logger.info("Starting create_derived_features step...")
    print("Starting create_derived_features step...")
    #print("dataset_dtypes: ", dataset.dtypes)
    print("dataset_shape: ", dataset.shape)
    print("lags: ", lags)
    
    try:
        
        # lag features for the target column
        for lag in range(1, lags+1):
            dataset['pedestrians_count_lag_'+str(lag)] = dataset['pedestrians_count'].shift(lag)
            
        # lead (and lag) features for the event datas
        # lead features are possible data leakage, but these here are not, cause it is known in advance if there is an event, holiday or weekday 
        for lag in range(1, lags+1):
            event_dataset['event_lag_'+str(lag)] = event_dataset['event'].shift(lag)
            event_dataset['holiday_lag_'+str(lag)] = event_dataset['holiday'].shift(lag)
            event_dataset['workday_lag_'+str(lag)] = event_dataset['workday'].shift(lag)
            
        for lead in range(1, lags+1):
            event_dataset['event_lead_'+str(lead)] = event_dataset['event'].shift(-lead)
            event_dataset['holiday_lead_'+str(lead)] = event_dataset['holiday'].shift(-lead)
            event_dataset['workday_lead_'+str(lead)] = event_dataset['workday'].shift(-lead)
        
        # lag features for the weather datas, keine lead features, da wir die Wetterdaten nicht in der Zukunft kennen
        #for lag in range(1, lags+1):
        #    dataset['temp_lag_'+str(lag)] = dataset['temp'].shift(lag)
        #    dataset['humidity_lag_'+str(lag)] = dataset['humidity'].shift(lag)
        #    dataset['precip_lag_'+str(lag)] = dataset['precip'].shift(lag)
            
        # chekc if lag und lead features are created
        #print("dataset_dtypes: ", dataset.dtypes)
        print("dataset_shape: ", dataset.shape)
        
        # fill the missing values of the column timestamp with the infos of the column date or datetime
        for i in range(len(dataset)):
            if pd.isnull(dataset['timestamp'].iloc[i]):
                if pd.isnull(dataset['date'].iloc[i]):
                    dataset['timestamp'].iloc[i] = dataset['datetime'].iloc[i]
                else:
                    dataset['timestamp'].iloc[i] = dataset['date'].iloc[i]
        
        
        # add time features: year, month, day, hour, weekday
        dataset['year'] = dataset['timestamp'].str.extract(r'(\d{4})').astype(int)
        dataset['month'] = dataset['timestamp'].str.extract(r'-(\d{2})-').astype(int)
        dataset['day'] = dataset['timestamp'].str.extract(r'-(\d{2})T').astype(int)
        dataset['hour'] = dataset['timestamp'].str.extract(r'T(\d{2})').astype(int)
        dataset['weekday'] = pd.to_datetime(dataset['date']).dt.day_name() # soll später one hot encoded werden
        
        #print("dataset_dtypes: ", dataset.dtypes)
        print("dataset_shape: ", dataset.shape)
        
        # need to drop the timestamp column, cause we dont need it anymore and it cant be fit_transformed by the pipeline
        #dataset.drop('timestamp', axis=1, inplace=True)
        #dataset.drop('date', axis=1, inplace=True)
        #dataset.drop('datetime', axis=1, inplace=True)
        # not here, cause we need it for the recursive forecasting in predictor step, we will drop it in the predictor step
        
        # conrol print
        for col in dataset.columns:
            # if datetype is object, print the column name
            if dataset[col].dtype == 'object':
                print(col)
                
        logger.info("Create derived features step successfully completed.")
        
        return dataset, event_dataset
    
    except Exception as e:
        logger.error(f"Error in create_derived_features step: {e}")