# missing external data

from zenml import step 
import pandas as pd
import sqlite3
from typing_extensions import Annotated
@step
def load_data() -> pd.DataFrame:
    """
    Load data from the 'data.db' SQLite database.
    We save the raw data (dataset) to the cache of zenml. What the cache does is that it stores the data and the results of the steps.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with columns 'location_name', 'pedestrians_count', and 'temperature'. #temperature, weather_condition
    """
    
    print("We are inside the load_data step.")
    
    # 1. Connect to the SQLite database
    connection = sqlite3.connect('data.db')
    
    # 2. Load the data from the 'data' table
    # we only need the location_id, pedestrians_count and timestamp cause the other columns have missing values or not needed
    wue_data = pd.read_sql('SELECT location_id, pedestrians_count, timestamp FROM data ORDER BY timestamp', connection)
    wue_data['date'] = wue_data['timestamp'].str.extract(r'(\d{4}-\d{2}-\d{2})')
    
    # 3. Load the weather data from the 'weather' table
    #weather_data = pd.read_sql('SELECT * FROM weather ORDER BY timestamp', connection)
    
    # 4. Load the event data from the 'events' table
    event_data = pd.read_sql('SELECT * FROM events ORDER BY date', connection)
    
    # 5. Merge the data on the timestamp column
    data = pd.merge(wue_data, event_data, on='date', how='left')
    data.sort_values(by=['timestamp', 'location_id'], inplace=True)
    
    # 6. Close the database connection
    connection.close()
    
    #print(data.head())
    #print(data.columns)

    return data
    