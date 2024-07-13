from zenml import step 
import pandas as pd
import sqlite3
from typing_extensions import Annotated
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def load_data() -> pd.DataFrame:
    """
    Load data from the 'data.db' SQLite database.
    We save the raw data (dataset) to the cache of zenml. What the cache does is that it stores the data and the results of the steps.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with columns 'location_name', 'pedestrians_count', and 'temperature'. #temperature, weather_condition
    """
    logger.info("Starting load_data step...")
    #print("We are inside the load_data step.")
    
    try:
        # 1. Connect to the SQLite database
        connection = sqlite3.connect('data.db')
        
        # 2. Load the data from the 'data' table
        # we only need the location_id, pedestrians_count and timestamp cause the other columns have missing values or not needed
        wue_data = pd.read_sql('SELECT location_id, pedestrians_count, timestamp FROM data ORDER BY timestamp', connection)
        wue_data['date'] = wue_data['timestamp'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        wue_data['datetime'] = wue_data['timestamp'].str.split('+').str[0]
        
        #print(wue_data.shape)
        
        # 3. Load the weather data from the 'weather' table
        weather_data = pd.read_sql('SELECT datetime, temp, humidity, precip FROM weather ORDER BY datetime', connection)
        
        #print(weather_data.shape)
        
        # 4. Load the event data from the 'events' table
        event_data = pd.read_sql('SELECT * FROM events ORDER BY date', connection)
        
        # 5. Merge the data on the timestamp column
        data = pd.merge(wue_data, event_data, on='date', how='left')
        #print(data.shape)
        data = pd.merge(data, weather_data, on='datetime', how='left')
        data.sort_values(by=['timestamp', 'location_id'], inplace=True)
        #print(data.shape)
        
        # noch kein dropen der Spalten, da wir diese noch beim nächsten Schritt benötigen
        
        # 6. Close the database connection
        connection.close()
        
        #print(data.head())
        #print(data.columns)
        
        logger.info("Load data step successfully completed.")

        return data
    
    except Exception as e:
        logger.error(f"Error in load_data step: {e}")
    