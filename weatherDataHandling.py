import pandas as pd
import sqlite3
import numpy as np
import importFiles as imf
data1=imf.handleImportOfPedestrianData()
weather_data1=imf.handleImportOfWeatherData()

def weatherHandling(data, weather_data):
    # Sample DataFrames
# Initialize the new columns in data with None
    for column in weather_data.columns:
        data[column] = None

# Create a mask to identify where the timestamp changes
    timestamp_changes = data['timestamp'].shift() != data['timestamp']

# Initialize the index for weather_data
    weather_data_index = 0

# Assign values from weather_data to data based on the timestamp changes
    for idx, change in timestamp_changes.items():
        if change and weather_data_index < len(weather_data):
            for column in weather_data.columns:
                data.at[idx, column] = weather_data.at[weather_data_index, column]
            weather_data_index += 1
        elif weather_data_index < len(weather_data):
            for column in weather_data.columns:
                data.at[idx, column] = data.at[idx - 1, column]

    return data

result_df = weatherHandling(data1, weather_data1)
print("results", result_df)