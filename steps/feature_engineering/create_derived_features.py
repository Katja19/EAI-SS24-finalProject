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
    dataset['event_lead_1'] = dataset['event'].shift(-1)
    dataset['event_lead_2'] = dataset['event'].shift(-2)
    
    dataset['holiday_lead_1'] = dataset['holiday'].shift(-1) 
    dataset['holiday_lead_2'] = dataset['holiday'].shift(-2)
    dataset['holiday_lag_1'] = dataset['holiday'].shift(1) 
    dataset['holiday_lag_2'] = dataset['holiday'].shift(2)
    
    dataset['workday_lead_1'] = dataset['workday'].shift(-1)
    dataset['workday_lead_2'] = dataset['workday'].shift(-2)
    dataset['workday_lag_1'] = dataset['workday'].shift(1)
    dataset['workday_lag_2'] = dataset['workday'].shift(2)
    
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
    
    
    
    #print("Derived features created inside the pipeline.")
    # print(dataset.head())
    # print(dataset.columns)
    # print(dataset.dtypes)
    
    return dataset