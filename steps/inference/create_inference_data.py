import pandas as pd
from zenml import step
from datetime import datetime, timedelta
import requests
from typing import Tuple
import io

def get_weather_forecast_24h(first_date: datetime, last_date: datetime) -> pd.DataFrame:
    """
    Get the weather forecast for the next 24 hours from the OpenWeatherMap API.
    Args:
        first_date (datetime): The first date of the next 24 hours.
        last_date (datetime): The last date of the next 24 hours.
        
    Returns:
        pd.DataFrame: A DataFrame containing the weather forecast for the next 24 hours.
    """

    # 1. Set up the API key and the base URL
    api_key = 'QPX3CJ476JEHWLUYT3TC559U8'
    
    # Define the endpoint URL and parameters
    base_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/w%C3%BCrzburg'
    params = {
        'unitGroup': 'metric',  # Use 'us' for Fahrenheit, 'metric' for Celsius
        'include': 'hours',  # Include hourly weather data
        'key': api_key,
        'contentType': 'csv'  # Request CSV format
    }
    # Construct the full API URL
    url = f"{base_url}/{first_date.date().strftime('%Y-%m-%d')}/{last_date.date().strftime('%Y-%m-%d')}"
    
    # Send the GET request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        
        # Read the response as a DataFrame
        weather_data_24h = pd.read_csv(io.StringIO(response.text))
        
        # dropping the columns that we don't need later, we want only to keep the coumns:datetime, temp, humidity, precip
        weather_data_24h = weather_data_24h[['datetime', 'temp', 'humidity', 'precip']]
        
    else:
        raise Exception(f"Error in get_weather_forecast_24h: {response.text}")
    
    # convert the csv data to a pandas dataframe
    weather_data_24h_df = pd.DataFrame(weather_data_24h)
    
    # now drop the rows that are not in the next 24 hours we want to predict
    first_date_str = first_date.strftime('%Y-%m-%dT%H:%M:%S')
    last_date_str = last_date.strftime('%Y-%m-%dT%H:%M:%S')
    weather_data_24h_df = weather_data_24h_df[(weather_data_24h_df['datetime'] >= first_date_str) & (weather_data_24h_df['datetime'] <= last_date_str)]
    
    return weather_data_24h_df
        
@step 
def create_inference_data(dataset_hist: pd.DataFrame, event_dataset:pd.DataFrame, lags:int, model_type:str) -> Tuple[pd.DataFrame, str, str]:
    
    # Get the latest timestamp of the pedestrian data in the format 'YYYY-MM-DDTHH:MM:SS'
    latest_datetime_wue_data = dataset_hist['timestamp'].max().split('+')[0]
    latest_datetime_wue_data = datetime.strptime(latest_datetime_wue_data, '%Y-%m-%dT%H:%M:%S')
    
    # Calculate the first and last date (both inclusive) of the next 24 hours we want to predict
    first_date = latest_datetime_wue_data + timedelta(hours=1)
    last_date = latest_datetime_wue_data + timedelta(hours=24)
    
    # Get a DataFrame with the next 24 hours weather forecast data
    weather_forecast_24h = get_weather_forecast_24h(first_date, last_date)
    
    # Get list of unique values from the column 'location_id'
    location_ids = dataset_hist['location_id'].unique()
    
    # Create three DataFrames for the three locations and add the location_id to the weather_forecast_24h
    weather_forecast_24h_1 = weather_forecast_24h.copy()
    weather_forecast_24h_1['location_id'] = location_ids[0]
    weather_forecast_24h_2 = weather_forecast_24h.copy()
    weather_forecast_24h_2['location_id'] = location_ids[1]
    weather_forecast_24h_3 = weather_forecast_24h.copy()
    weather_forecast_24h_3['location_id'] = location_ids[2]
    
    # Merge the three DataFrames into one
    weather_forecast_24h = pd.concat([weather_forecast_24h_1, weather_forecast_24h_2, weather_forecast_24h_3], axis=0)
    
    # Concatenate dataset_hist and weather_forecast_24h vertically
    inference_data = pd.concat([dataset_hist, weather_forecast_24h], axis=0)
    
    # Extract year, month, day, hour, and weekday from the datetime column and fill NaN values
    inference_data['year'] = inference_data['year'].fillna(inference_data['datetime'].str[:4].astype(int))
    inference_data['month'] = inference_data['month'].fillna(inference_data['datetime'].str[5:7].astype(int))
    inference_data['day'] = inference_data['day'].fillna(inference_data['datetime'].str[8:10].astype(int))
    inference_data['hour'] = inference_data['hour'].fillna(inference_data['datetime'].str[11:13].astype(int))
    inference_data['weekday'] = inference_data['weekday'].fillna(pd.to_datetime(inference_data['datetime']).dt.day_name())
    
    # Create lag features for the weather data
    for lag in range(1, lags+1):
        inference_data[f'temp_lag_{lag}'] = inference_data['temp'].shift(lag)
        inference_data[f'humidity_lag_{lag}'] = inference_data['humidity'].shift(lag)
        inference_data[f'precip_lag_{lag}'] = inference_data['precip'].shift(lag)
        
    # Create lag features for pedestrians_count
    for lag in range(1, lags+1):
        inference_data[f'pedestrians_count_lag_{lag}'] = inference_data['pedestrians_count'].shift(lag)
    
    # Create lag and lead features for the event data
    for lag in range(1, lags+1):
        event_dataset[f'event_lag_{lag}'] = event_dataset['event'].shift(lag)
        event_dataset[f'holiday_lag_{lag}'] = event_dataset['holiday'].shift(lag)
        event_dataset[f'workday_lag_{lag}'] = event_dataset['workday'].shift(lag)
        
    for lead in range(1, lags+1):
        event_dataset[f'event_lead_{lead}'] = event_dataset['event'].shift(-lead)
        event_dataset[f'holiday_lead_{lead}'] = event_dataset['holiday'].shift(-lead)
        event_dataset[f'workday_lead_{lead}'] = event_dataset['workday'].shift(-lead)
        
    # Merge event data with inference data on the date column
    inference_data = pd.merge(inference_data, event_dataset, on='date', how='left')
    
    # Filter inference_data to get only the rows that are in the next 24 hours
    first_date_str = first_date.strftime('%Y-%m-%dT%H:%M:%S')
    last_date_str = last_date.strftime('%Y-%m-%dT%H:%M:%S')
    inference_data = inference_data[(inference_data['datetime'] >= first_date_str) & (inference_data['datetime'] <= last_date_str)]
    
    # drop columns that are not needed
    # drop all columns that have only NaN values in its columns
    for column in inference_data.columns:
        if inference_data[column].isnull().all():
            inference_data.drop(column, axis=1, inplace=True)
    
    # Save the inference data as a CSV file
    inference_data.to_csv("inference_data_original_1.csv", index=False)
    return inference_data, first_date_str, last_date_str