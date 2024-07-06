from zenml import step
import pandas as pd
from typing_extensions import Annotated
from sklearn.linear_model import LinearRegression
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

@step
def feature_engineering(X_train: pd.DataFrame,X_test:pd.DataFrame)-> Tuple[Annotated[pd.DataFrame, "X_train_encoded"], Annotated[pd.DataFrame, "X_test_encoded"]]:
    """
    This step is responsible for encoding the categorical variables in the dataset.
    At the moment we drop all other features

    Parameters:
    X_train (pd.DataFrame): The input features for training the model.
    X_test (pd.DataFrame): The input features for testing the model.

    Returns:
    Tuple[Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"]]: A tuple containing the encoded training and testing data.
    """
    
    print("Data in feaure engenierien:")
    print(X_train.head())
    
    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_train_values = pd.DataFrame(encoder.fit_transform(X_train[["location_name"]]), index=X_train.index, columns=encoder.get_feature_names_out())
    encoded_test_values = pd.DataFrame(encoder.transform(X_test[["location_name"]]), index=X_test.index, columns=encoder.get_feature_names_out())
    return encoded_train_values, encoded_test_values