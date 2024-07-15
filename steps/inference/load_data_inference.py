from zenml import step 
import pandas as pd
import sqlite3
from typing import Tuple

@step
def load_data_inference() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from the 'data.db' SQLite database.
    We save the raw data (dataset) to the cache of zenml. What the cache does is that it stores the data and the results of the steps.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with columns 'location_name', 'pedestrians_count', and 'temperature'. #temperature, weather_condition
    """
    
    #print("We are inside the load_data step.")
    
    # 1. Connect to the SQLite database
    connection = sqlite3.connect('data.db')
    
    # 2. Load the data from the 'data' table
    wue_data = pd.read_sql('SELECT location_id, pedestrians_count, timestamp FROM data ORDER BY timestamp', connection)
    wue_data['date'] = wue_data['timestamp'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    wue_data['datetime'] = wue_data['timestamp'].str.split('+').str[0]
    
    # 3. Load the weather data from the 'weather' table
    weather_data = pd.read_sql('SELECT datetime, temp, humidity, precip FROM weather ORDER BY datetime', connection)
    
    # 4. Load the event data from the 'events' table
    event_data = pd.read_sql('SELECT * FROM events ORDER BY date', connection)
    
    # 5. Merge the data on the timestamp column but dont drop rows with missing values
    #data = pd.merge(wue_data, event_data, on='date', how='outer')
    data = pd.merge(wue_data, weather_data, on='datetime', how='left')
    data.sort_values(by=['timestamp', 'location_id'], inplace=True)

    # noch kein dropen der Spalten, da wir diese noch beim nächsten Schritt benötigen
    
    # 6. Close the database connection
    connection.close()
    
    # print("Data loaded successfully.")
    # print("data.isnull().sum()")
    # for col in data.columns:
    #     print(col, data[col].isnull().sum())
    # print("data.shape: ", data.shape)
    
    # print("event_data.isnull().sum()")
    # for col in event_data.columns:
    #     print(col, event_data[col].isnull().sum())
    # print("event_data.shape: ", event_data.shape)

    return data, event_data