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
    connection = sqlite3.connect('data.db')
    
    dataset = pd.read_csv("https://opendata.wuerzburg.de/api/explore/v2.1/catalog/datasets/passantenzaehlung_stundendaten/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",delimiter=";")
    dataset.drop(["min_temperature","details","GeoShape","GeoPunkt"],axis=1,inplace=True)
    try:
        existing_data = pd.read_sql('SELECT * FROM data', connection)
    except:
        existing_data = pd.DataFrame(columns=["location_name", "pedestrians_count", "temperature"])
    merged_data = pd.concat([existing_data, dataset]).drop_duplicates().reset_index(drop=True)
    merged_data.to_sql('data', connection, if_exists='replace', index=False)
    connection.commit()
    connection.close()
