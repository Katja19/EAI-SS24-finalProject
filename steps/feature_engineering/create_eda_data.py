from zenml import step
import pandas as pd
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def create_eda_data(X_train:pd.DataFrame,X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series, X_train_eda_date_infos:pd.DataFrame, X_test_eda_date_infos:pd.DataFrame):
    
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    print("Starting create_eda_data step...")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    #ckeck for nan values
    print(f"X_train nan values: {X_train.isnull().sum().sum()}")
    print(f"X_test nan values: {X_test.isnull().sum().sum()}")
    print(f"y_train nan values: {y_train.isnull().sum()}")
    print(f"y_test nan values: {y_test.isnull().sum()}")
    
    # Reset index to ensure proper concatenation
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train_eda_date_infos = X_train_eda_date_infos.reset_index(drop=True)
    X_test_eda_date_infos = X_test_eda_date_infos.reset_index(drop=True)
    
    # Add the date information back to the dataframes
    X_train = pd.concat([X_train, X_train_eda_date_infos], axis=1) 
    X_test = pd.concat([X_test, X_test_eda_date_infos], axis=1)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Merge the X_train and y_train dataframes
    eda_train = pd.concat([X_train, y_train], axis=1)
    eda_test = pd.concat([X_test, y_test], axis=1)

    print(f"eda_train shape: {eda_train.shape}")
    print(f"eda_test shape: {eda_test.shape}")

    # Concatenate train and test dataframes
    eda_df = pd.concat([eda_train, eda_test], axis=0)
    print(f"eda_df shape: {eda_df.shape}")
    
    # Save to CSV
    eda_df.to_csv("data/processed/eda_data.csv", index=False)
    print("EDA data saved successfully.")