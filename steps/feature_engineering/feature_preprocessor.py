# fertig

from zenml import step
from sklearn.pipeline import Pipeline
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple

@step
def feature_preprocessor(pipeline:Pipeline,X_train:pd.DataFrame,X_test:pd.DataFrame)-> Tuple[Annotated[pd.DataFrame,"X_train_preprocessed"],
                                                                                             Annotated[pd.DataFrame,"X_test_preprocessed"],
                                                                                             Annotated[Pipeline,"pipeline"]]:
    
    """
        Perform feature engineering on the X data and return the preprocessed data and the fitted pipeline.
        Here are saved the preprocessed data (X_train_preprocessed, X_test_preprocessed) and the fitted pipeline.
    """
    
    # 1. Fit the pipeline on the training data to learn the necessary transformations and transform the training data and the test data
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # 2. Get the feature names after the transformation
    cat_features_after_encoding = pipeline.named_steps['preprocessing'].transformers_[1][1].named_steps['cat_encoder'].get_feature_names_out(X_train.select_dtypes(include=['object']).columns)
    all_features = list(X_train.select_dtypes(exclude=['object']).columns) + list(cat_features_after_encoding)
    
    # 3. Create DataFrames with the transformed data
    X_train_preprocessed = pd.DataFrame(X_train_transformed,columns=all_features)
    X_test_preprocessed = pd.DataFrame(X_test_transformed,columns=all_features)
    
    return X_train_preprocessed,X_test_preprocessed,pipeline
    