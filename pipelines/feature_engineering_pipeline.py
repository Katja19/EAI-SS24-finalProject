from steps import update_data, load_data, split_data, create_derived_features, create_preprocessing_pipeline, feature_preprocessor, create_eda_data
from zenml import pipeline
import pandas as pd
import os

import logging

"""
    This pipeline will update the data and will perform feature engineering on the data.
    We will also need feature engineering during the inference later on.
    In this pipeline, we will save
    - the raw data (X_train, X_test, y_train, y_test, input_data)
    - the preprocessed data (X_train_preprocessed, X_test_preprocessed, y_train_encoded, y_test_encoded)
    - the prepro_pipeline, which is the pipeline that was used to preprocess the data
"""

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# The cache is used to store the data and the results of the steps, so we dont set: @pipeline#(enable_cache=False)
@pipeline
def feature_engineering_pipeline(model_variant:str, model_type:str, lags:int, trials:int):
    """
        Pipeline to update the data and perform feature engineering on the data.
    """
    logger.info("Starting feature engineering pipeline...")
    
    
    #print("We are inside the feature engineering pipeline.")
    # 0. ensure the execution order of the steps
    load_data.after(update_data)
    create_derived_features.after(load_data)
    split_data.after(create_derived_features)
    create_preprocessing_pipeline.after(split_data)
    feature_preprocessor.after(create_preprocessing_pipeline)
    create_eda_data.after(feature_preprocessor)
    
    # 1. update the data
    update_data()
    logger.info("Data updated.")
    #print(f"Data updated.")

    # 2. load the data (including the weather and traffic data the end)
    dataset = load_data()
    #print("Data loaded.")
    #print(type(dataset))

    # 2.5 create derived features, we will loose the first num_lags rows
    dataset = create_derived_features(dataset, lags)
    
    # 3. split the data into training and test data
    X_train,X_test,y_train,y_test,X_train_eda_date_infos, X_test_eda_date_infos = split_data(dataset,"pedestrians_count")
    #X_train,X_test,y_train,y_test = split_data(dataset,"pedestrians_count")
    # print("Data splitted.")
    # print(type(X_train))
    
    # 4. create a preprocessing pipeline for the feature engineering
    # # it includes the steps for feature transformation (imputation, scaling, encoding, etc.)
    prepro_pipeline = create_preprocessing_pipeline(dataset,"pedestrians_count")
    # print("Preprocessing pipeline created")
    
    # 5. perform feature engineering on the X data and return the preprocessed data and 
    # # Now the pipeline is fitted on the training data to learn the necessary transformations, 
    # # that will be applied to the test data later on.
    X_train,X_test,fitted_pipeline = feature_preprocessor(prepro_pipeline,X_train,X_test, model_variant, model_type, lags, trials) 

    # 6. save the preprocessed data as a csv file for EDA
    # Create directory if not exists
    #X_train_eda_date_infos = X_train[["Year", "Month", "Day", "Hour"]].copy()
    #X_test_eda_date_infos = X_test[["Year", "Month", "Day", "Hour"]].copy()
    create_eda_data(X_train,X_test,y_train,y_test,X_train_eda_date_infos, X_test_eda_date_infos)
    
    logger.info("Feature engineering pipeline successfully completed.")
    
    return fitted_pipeline
    
    