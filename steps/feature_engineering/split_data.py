# fertig

import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from typing import Tuple


@step
def split_data(dataset:pd.DataFrame, label: str) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]]:
    """
    Splits a dataset into training and testing sets.

    This function takes a pandas DataFrame and a specified label column, then
    divides the data into training and testing subsets. The splitting process does
    not shuffle the data, which preserves the original ordering in the training
    and testing sets.
    We save the raw data (X_train, X_test, y_train, y_test) to the cache of zenml.
    """
    #print("We are inside the split_data step.")
    
    
    # 1. Split the data into features X and target(=label) y
    X = dataset.drop(label,axis=1)
    y = dataset[label]
    
    # 2. Split the data into training and testing sets, without shuffling the data and with a test size of 20%
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    # print the shapes of the training and testing data
    # print("Shapes of the training and testing data:")
    # print(dataset.shape)
    # print(X_train.dtypes)
    # print(X_train.shape)
    # print(X_test.dtypes)
    # print(X_test.shape)
    # print(y_train.dtypes)
    # print(y_train.shape)
    # print(y_test.dtypes)
    # print(y_test.shape)

    
    
    
    return X_train,X_test,y_train,y_test