import sqlite3
import pandas as pd
from zenml import step
from datetime import datetime
import sqlite3
import pandas as pd
import requests
import io


@step 
def create_inference_data(lags) -> pd.DataFrame:
    
    # 0. make a connection to the SQLite database
    connection = sqlite3.connect('data.db')
    
    
    # 1. load the pedestrian data from the SQLite database from the table 'data'
    wue_data = pd.read_sql('SELECT location_id, pedestrians_count, timestamp FROM data ORDER BY timestamp', connection)
    # for merging the dataframes
    wue_data['date'] = wue_data['timestamp'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    wue_data['datetime'] = wue_data['timestamp'].str.split('+').str[0]
    
    # 2. load the event data from the SQLite database from the table 'events'
    event_data = pd.read_sql('SELECT * FROM events ORDER BY date', connection)
    
    # 3. load the weather data from the SQLite database from the table 'weather'
    weather_data = pd.read_sql('SELECT datetime, temp, humidity, precip FROM weather ORDER BY datetime', connection)
    
    # 4. get the latest timestamp of the pedestrian data in the str format 'YYYY-MM-DDTHH:MM:SS'
    latest_datetime_wue_data = wue_data['timestamp'].max().split('+')[0]
    latest_datetime_wue_data = datetime.strptime(latest_datetime_wue_data, '%Y-%m-%dT%H:%M:%S')
    
    # 5. calculate first and last date of the next 24 hours that we want to predict
    first_date = latest_datetime_wue_data + datetime.timedelta(hours=1)
    last_date = latest_datetime_wue_data + datetime.timedelta(hours=24)
    
    # 5 get a dataframe with the next 24 hours weahter forecast data
    #weather_forecast_24h = get_weather_forecast_24h(first_date, last_date)
    
    # 6. merge the weather datasets
    
    # 7. merge the event datasets
    
    
    
    
    
    
    
    
    inference_data = pd.DataFrame()
    
    return inference_data