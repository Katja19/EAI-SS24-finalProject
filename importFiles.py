import pandas as pd
import sqlite3
import numpy as np
def handleImportOfPedestrianData():
    connection = sqlite3.connect('data.db')
    data = pd.read_sql('SELECT * FROM data ORDER BY timestamp', connection)
    return data

def handleImportOfWeatherData():
    weather_data=pd.read_csv('wuerzburg.csv')
    return weather_data

