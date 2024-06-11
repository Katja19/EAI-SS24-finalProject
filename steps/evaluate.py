from zenml import step
import pandas as pd
from typing import Annotated
from sklearn.base import RegressorMixin
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Tuple

@step
def evaluate(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "mae"], Annotated[np.ndarray, "predictions"]]:
    """
    Evaluate the performance of a regression model on the test data.

    Parameters:
    model (RegressorMixin): The trained regression model.
    X_test (pd.DataFrame): The input features of the test data.
    y_test (pd.Series): The target values of the test data.

    Returns:
    Tuple[Annotated[float, "mae"], Annotated[np.ndarray, "predictions"]]: A tuple containing the mean absolute error (mae) and the predicted values.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae, predictions