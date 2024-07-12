# missing external data

import sqlite3
import pandas as pd
from zenml import step
from datetime import datetime
import sqlite3
import pandas as pd
import requests
import io

@step 
def update_data():
    """
    Updates the data in the 'data.db' SQLite database with the latest data from the provided URL.

    This function connects to the 'data.db' database, retrieves the latest data from the specified URL,
    removes unnecessary columns from the dataset, and replaces the existing data in the 'data' table
    with the updated dataset. Finally, it commits the changes and closes the database connection.
    """
    
    print("We are inside the update_data step.")
    
    # 1. Connect to the SQLite database
    connection = sqlite3.connect('data.db')
    
    ############################################################
    # 2. City Wuerzburg pedestrian data
    # 2.1 Retrieve the latest data from the specified URL
    dataset = pd.read_csv("https://opendata.wuerzburg.de/api/explore/v2.1/catalog/datasets/passantenzaehlung_stundendaten/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",delimiter=";")
    
    # 2.2 Remove unnecessary columns from the dataset
    dataset.drop(["min_temperature","details","GeoShape","GeoPunkt"],axis=1,inplace=True)
    
   # 2.3 Attempt to read the existing data from the 'data' table, create an empty DataFrame 
   # if the table does not exist
    try:
        existing_data = pd.read_sql('SELECT * FROM data', connection)
    except:
        existing_data = pd.DataFrame(columns=["location_name", "pedestrians_count", "temperature"])
    
    # 2.4 Merge the existing data with the updated dataset and remove duplicates
    merged_data = pd.concat([existing_data, dataset]).drop_duplicates().reset_index(drop=True)
    
    # 2.5 Replace the existing data in the 'data' table with the merged dataset
    merged_data.to_sql('data', connection, if_exists='replace', index=False)
    
    ############################################################
    # 3. historical weather data for Wuerzburg
    # 3.1 get the lastest datetime in the 'weather' table (date)
    weather_temp = pd.read_sql('SELECT * FROM weather ORDER BY datetime', connection)
    latest_daytime = weather_temp['datetime'].max()
    latest_date= latest_daytime.split('T')[0]
    current_datetime_rounded = (datetime.now()).replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:%S')
    current_date = current_datetime_rounded.split('T')[0]
    
    # 3.2 Retrieve the latest weather data from the specified URL
    # Replace with your Visual Crossing API key
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
    url = f"{base_url}/{latest_date}/{current_date}"

    # Send the GET request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Read the response as a DataFrame
        weather_data = pd.read_csv(io.StringIO(response.text))
        print(weather_data.head())
        print(weather_data.shape)
        
        # save the weather data to the 'weather' table
        try:
            existing_weather_data = pd.read_sql('SELECT * FROM weather', connection)
        except:
            existing_weather_data = pd.DataFrame(columns=weather_data.columns)
        
        # Filter the weather data to get only the rows that are between the latest date and the current date both excluded
        weather_data = weather_data[(weather_data['datetime'] > latest_daytime) & (weather_data['datetime'] < current_datetime_rounded)]
        
        # Merge the existing weather data with the updated dataset and remove duplicates
        merged_weather_data = pd.concat([existing_weather_data, weather_data]).drop_duplicates().reset_index(drop=True) # if duplicates are found, the first occurrence is kept
        
        # Replace the existing weather data in the 'weather' table with the merged dataset
        merged_weather_data.to_sql('weather', connection, if_exists='replace', index=False)
    else:
        print(f"Failed to retrieve weather data. Status code: {response.status_code}")
    
    
    # 4. Commit the changes and close the database connection
    connection.commit()
    connection.close()
