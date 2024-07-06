from zenml import step
import pandas as pd
from typing_extensions import Annotated
import pmdarima as pm
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error
from typing import Tuple



@step
def create_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Annotated[RegressorMixin, "model"], Annotated[float, "mae_in_sample"]]:
    """
    Create a linear regression model and train it on the training data.

    Parameters:
    X_train (pd.DataFrame): The input features for training the model.
    y_train (pd.Series): The target variable for training the model.

    Returns:
    Tuple[Annotated[RegressorMixin, "model"], Annotated[float, "mae_in_sample"]]: A tuple containing the trained model and the mean absolute error (MAE) in the training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train) # Train the model
    mae_in_sample = mean_absolute_error(y_train, model.predict(X_train))
    return model, mae_in_sample
