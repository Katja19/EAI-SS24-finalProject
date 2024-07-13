from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing_extensions import Annotated
import logging


"""
    Create a preprocessing pipeline for the feature engineering.
    Here we impute, scale, and encode the features of the data.
"""
   
# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
@step
def create_preprocessing_pipeline(dataset:pd.DataFrame,target:str) -> Pipeline:
    """_summary_

    Args:
        dataset (pd.DataFrame): _description_
        target (str): _description_

    Returns:
        Pipeline: _description_
    """
    
    logger.info("Starting create_preprocessing_pipeline step...")
    try:
        #print("Inside the create_preprocessing_pipeline step.")
        #print("The target column is: ", target)
        #print("The dataset columns are: ", dataset.columns)
        
        # 1. Define the numerical and categorical columns excluding the target column
        cat_columns = dataset.select_dtypes(include=['object']).columns
        #print("Categorical columns: ", cat_columns)
        
        num_columns = dataset.select_dtypes(exclude=['object']).columns
        #print("Numerical columns: ", num_columns)
        num_columns = num_columns.drop(target)
            
        # 2. Define the numerical pipeline with SimpleImputer for missing values and StandardScaler for scaling
        num_pipeline = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('num_scaler', StandardScaler())
        ])
            
        # 3.Define the categorical pipeline with SimpleImputer for missing values and OneHotEncoder for encoding
        cat_pipeline = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_encoder', OneHotEncoder())
        ])
            
        # 4. Combine numerical and categorical pipelines using ColumnTransformer
        preprocessing = ColumnTransformer([
            ('num', num_pipeline, num_columns),
            ('cat', cat_pipeline, cat_columns)
        ])
            
        # 5. Create a final pipeline that includes the preprocessing steps
        prepro_pipeline = Pipeline([
            ('preprocessing', preprocessing)
        ])
        
        logger.info("Preprocessing pipeline step successfully completed.")
            
        return prepro_pipeline
    
    except Exception as e:
        logger.error(f"Error in create_preprocessing_pipeline step: {e}")