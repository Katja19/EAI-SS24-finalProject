# missing external data

import sqlite3
import pandas as pd
from zenml import step
import sqlite3
import pandas as pd

@step 
def update_data():
    """
    Updates the data in the 'data.db' SQLite database with the latest data from the provided URL.

    This function connects to the 'data.db' database, retrieves the latest data from the specified URL,
    removes unnecessary columns from the dataset, and replaces the existing data in the 'data' table
    with the updated dataset. Finally, it commits the changes and closes the database connection.
    """
    
    # 1. Connect to the SQLite database
    connection = sqlite3.connect('data.db')
    
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
    
    # TODO: Ungefähr so könnte der Code aussehen, um die Wetterdaten zu aktualisieren
    # 3. historical weather data for Wuerzburg
    # 3.1 Retrieve the latest weather data from the specified URL
    # 3.2 Remove unnecessary columns from the weather dataset
    # 3.3 Attempt to read the existing weather data from the 'weather' table, create an empty DataFrame
    # 3.4 Merge the existing weather data with the updated weather dataset and remove duplicates
    # 3.5 Replace the existing weather data in the 'weather' table with the merged weather dataset
    
    # TODO: Ungefähr so könnte der Code aussehen, um die Feitagdaten zu aktualisieren
    # 4. historical traffic data for Wuerzburg
    # 4.1 Retrieve the latest traffic data from the specified URL
    # 4.2 Remove unnecessary columns from the traffic dataset
    # 4.3 Attempt to read the existing traffic data from the 'traffic' table, create an empty DataFrame
    # 4.4 Merge the existing traffic data with the updated traffic dataset and remove duplicates
    # 4.5 Replace the existing traffic data in the 'traffic' table with the merged traffic dataset
    
    # 5. Commit the changes and close the database connection
    connection.commit()
    connection.close()
