from zenml import step 
import pandas as pd
import sqlite3
from typing_extensions import Annotated
@step
def load_data() -> Annotated[pd.DataFrame,"dataset"]:
    """
    Load data from the 'data.db' SQLite database.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with columns 'location_name', 'pedestrians_count', and 'temperature'.
    """
    connection = sqlite3.connect('data.db')
    data = pd.read_sql('SELECT location_name, pedestrians_count, temperature FROM data ORDER BY timestamp', connection)
    return data
    