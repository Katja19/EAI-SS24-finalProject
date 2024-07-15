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
    
    print("Starting create_inference_data step...")
    print("dataset_hist.columns")
    print(dataset_hist.columns)
    
    # get a dataframe with the next 24 hours weahter forecast data
    # 1 get the latest timestamp of the pedestrian data in the str format 'YYYY-MM-DDTHH:MM:SS'
    latest_datetime_wue_data = dataset_hist['timestamp'].max().split('+')[0]
    latest_datetime_wue_data = datetime.strptime(latest_datetime_wue_data, '%Y-%m-%dT%H:%M:%S')
    
    # 2 calculate first and last date (both inclusive) of the next 24 hours that we want to predict
    first_date = latest_datetime_wue_data + timedelta(hours=1)
    last_date = latest_datetime_wue_data + timedelta(hours=24)
    
    # 3 get a dataframe with the next 24 hours weahter forecast data
    weather_forecast_24h = get_weather_forecast_24h(first_date, last_date)
    
    # get list of unique values from the column 'location_id'
    location_ids = dataset_hist['location_id'].unique()
    # create thre dataframes for the three locations and add the location_id to the weather_forecast_24h
    weather_forecast_24h_1 = weather_forecast_24h.copy()
    weather_forecast_24h_1['location_id'] = location_ids[0]
    weather_forecast_24h_2 = weather_forecast_24h.copy()
    weather_forecast_24h_2['location_id'] = location_ids[1]
    weather_forecast_24h_3 = weather_forecast_24h.copy()
    weather_forecast_24h_3['location_id'] = location_ids[2]
    
    # merge the three dataframes to one
    weather_forecast_24h = pd.concat([weather_forecast_24h_1, weather_forecast_24h_2, weather_forecast_24h_3], axis=0)
    
    print("weather_forecast_24h.columns")
    print(weather_forecast_24h.columns)
    
    # 5 merge die weather_forecast_24h with the dataset_hist on the timestamp column and 
    # fill the missing values with nan, cause they will be filled during the recursive forecasting with the model
    #inference_data = pd.merge(dataset_hist, weather_forecast_24h, on=['datetime', 'location_id'], how='outer', axis=0) # vertical concatenation (untereinander)
    inference_data = pd.concat([dataset_hist, weather_forecast_24h], axis=0) # Vertical concatenation
    
    
    print("inference_data.columns")
    print(inference_data.columns)
    
    # 4 create lag features for the weather data
    for lag in range(1, lags+1):
        inference_data['temp_lag_'+str(lag)] = inference_data['temp'].shift(lag)
        inference_data['humidity_lag_'+str(lag)] = inference_data['humidity'].shift(lag)
        inference_data['precip_lag_'+str(lag)] = inference_data['precip'].shift(lag)
    
    # 6 filter the inference_data to get only the rows that are in the next 24 hours
    first_date_str = first_date.strftime('%Y-%m-%d') + 'T' + first_date.strftime('%H:%M:%S')
    last_date_str= last_date.strftime('%Y-%m-%dT%H:%M:%S')
    inference_data = inference_data[(inference_data['datetime'] >= first_date_str) & (inference_data['datetime'] <= last_date_str)]
    
    # 7 add event data to the inference data
    inference_data = pd.merge(inference_data, event_dataset, on='date', how='left')
    
    #controll
    print(inference_data.isnull().sum())
    
    # saving the inference data as an csv file
    # spit the first_date_str and last_date_str by 'T' and take the first part
    first_date_day = first_date_str.split('T')[0]
    first_date_hour = first_date_str.split('T')[1]
    last_date_day = last_date_str.split('T')[0]
    last_date_hour = last_date_str.split('T')[1]
    # get the fist two chars
    first_date_hour = first_date_hour[:2]
    last_date_hour = last_date_hour[:2]
    
    inference_data.to_csv(f"data/inference_{model_type}_{first_date_day}_{first_date_hour}_to_{last_date_day}_{last_date_hour}.csv", index=False)
    
    return inference_data, first_date_str, last_date_str