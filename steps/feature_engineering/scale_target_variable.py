import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np

@step
def scale_target_variable(y_train:pd.Series,y_test:pd.Series) -> Tuple[Annotated[pd.Series,"y_train_scaled"],
                                                                       Annotated[pd.Series,"y_test_scaled"]]:
    """
    Scales the target variable using logarithmic transformation and StandardScaler.
    """

    # Logarithmic transformation
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Standard scaling
    scaler = StandardScaler()
    y_train_scaled = pd.Series(scaler.fit_transform(y_train_log.values.reshape(-1, 1)).flatten())
    y_test_scaled = pd.Series(scaler.transform(y_test_log.values.reshape(-1, 1)).flatten())

    return y_train_scaled, y_test_scaled