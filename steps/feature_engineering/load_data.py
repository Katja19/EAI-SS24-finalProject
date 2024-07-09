# missing external data

from zenml import step 
import pandas as pd
import sqlite3
from typing_extensions import Annotated
@step
def load_data() -> Annotated[pd.DataFrame,"dataset"]:
    """
    Load data from the 'data.db' SQLite database.
    We save the raw data (dataset) to the cache of zenml. What the cache does is that it stores the data and the results of the steps.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with columns 'location_name', 'pedestrians_count', and 'temperature'.
    """
    
    # 1. Connect to the SQLite database
    connection = sqlite3.connect('data.db')
    
    # 2. Load the data from the 'data' table
    # we only need the location_id, pedestrians_count and timestamp cause the other columns have missing values or not needed
    data = pd.read_sql('SELECT location_id, pedestrians_count, timestamp, temperature, weather_condition FROM data ORDER BY timestamp', connection)
    # we getting temperature and weather_condition for now cause we dont have the weather data yet
    # but we will need to impute missing values for temperature and weather_condition later on 
    
    # 3. Load the weather data from the 'weather' table
    #weather_data = pd.read_sql('SELECT * FROM weather ORDER BY timestamp', connection)
    
    # 4. Load the traffic data from the 'traffic' table
    
    # 5. Merge the data on the timestamp column
    #data = pd.merge(data, weather_data, on='timestamp', how='left')
    
    # 6. Close the database connection
    print(data.head())
    print(data.columns)

    return data
    