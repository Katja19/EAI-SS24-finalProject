from zenml import step
import pandas as pd
import numpy as np
from typing_extensions import Annotated

@step
def create_derived_features(dataset:pd.DataFrame, lags:int) -> Annotated[pd.DataFrame,"dataset"]:
    """
    Create derived features from the dataset.
    Here we create lag features from the target column and
    time-based features.
    """
    
    #print("We are inside the create_derived_features step.")
    #print(dataset.head(10))
    
    # lag features for the target column
    for lag in range(1, lags+1):
        dataset['pedestrians_count_lag_'+str(lag)] = dataset['pedestrians_count'].shift(lag)
        
    # lead (and lag) features for the event datas
    # lead features are possible data leakage, but these here are not, cause it is known in advance if there is an event, holiday or weekday 
    for lag in range(1, lags+1):
        dataset['event_lag_'+str(lag)] = dataset['event'].shift(lag)
        dataset['holiday_lag_'+str(lag)] = dataset['holiday'].shift(lag)
        dataset['workday_lag_'+str(lag)] = dataset['workday'].shift(lag)
        
    for lead in range(1, lags+1):
        dataset['event_lead_'+str(lead)] = dataset['event'].shift(-lead)
        dataset['holiday_lead_'+str(lead)] = dataset['holiday'].shift(-lead)
        dataset['workday_lead_'+str(lead)] = dataset['workday'].shift(-lead)
    
    # lag features for the weather datas, keine lead features, da wir die Wetterdaten nicht in der Zukunft kennen
    for lag in range(1, lags+1):
        dataset['temp_lag_'+str(lag)] = dataset['temp'].shift(lag)
        dataset['humidity_lag_'+str(lag)] = dataset['humidity'].shift(lag)
        dataset['precip_lag_'+str(lag)] = dataset['precip'].shift(lag)
    
    # fill null values with backward fill
    #df.bfill(inplace=True)
    # drop rows with null values
    dataset.dropna(inplace=True)
    
    # add time features: year, month, day, hour, weekday
    dataset['year'] = dataset['timestamp'].str.extract(r'(\d{4})').astype(int)
    dataset['month'] = dataset['timestamp'].str.extract(r'-(\d{2})-').astype(int)
    dataset['day'] = dataset['timestamp'].str.extract(r'-(\d{2})T').astype(int)
    dataset['hour'] = dataset['timestamp'].str.extract(r'T(\d{2})').astype(int)
    dataset['weekday'] = pd.to_datetime(dataset['date']).dt.day_name() # soll später one hot encoded werden
    
    # need to drop the timestamp column, cause we dont need it anymore and it cant be fit_transformed by the pipeline
    dataset.drop('timestamp', axis=1, inplace=True)
    dataset.drop('date', axis=1, inplace=True)
    dataset.drop('datetime', axis=1, inplace=True)
    
    #print("Derived features created inside the pipeline.")
    # print(dataset.head())
    # print(dataset.columns)
    # print(dataset.dtypes)
    
    return dataset