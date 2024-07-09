from steps import update_data, load_data, split_data, create_preprocessing_pipeline, feature_preprocessor
from zenml import pipeline

"""
    This pipeline will update the data and will perform feature engineering on the data.
    We will also need feature engineering during the inference later on.
    In this pipeline, we will save
    - the raw data (X_train, X_test, y_train, y_test, input_data)
    - the preprocessed data (X_train_preprocessed, X_test_preprocessed, y_train_encoded, y_test_encoded)
    - the prepro_pipeline, which is the pipeline that was used to preprocess the data
"""

# enable_cache=False means that the pipeline will not use the cache
# The cache is used to store the data and the results of the steps
# If we set enable_cache=False, the steps of the pipeline will not be able to use the cache
@pipeline#(enable_cache=False) so we dont need to set enable_cache=False
def feature_engineering_pipeline():
    """
        Pipeline to update the data and perform feature engineering on the data.
    """
    # 0. ensure that the update_data step is executed before the load_data step
    load_data.after(update_data)
    
    # 1. update the data
    update_data()
    
    # 2. load the data
    dataset = load_data()
    
    # 3. split the data into training and test data
    X_train,X_test,y_train,y_test = split_data(dataset,"pedestrians_count")
    
    # OLD: 4. perform feature engineering on the X data
    #X_train,X_test = feature_engineering(X_train,X_test)
    
    # NEW: 4. create a preprocessing pipeline for the feature engineering
    # it includes the steps for feature transformation (imputation, scaling, encoding, etc.)
    prepro_pipeline = create_preprocessing_pipeline()
    
    # NEW: 5. perform feature engineering on the X data and return the preprocessed data and 
    # Now the pipeline is fitted on the training data to learn the necessary transformations, 
    # that will be applied to the test data later on.
    X_train,X_test,fitted_pipeline = feature_preprocessor(prepro_pipeline,X_train,X_test) 
    
    # NEW: 6. encode the y data not needed anymore, cause it is a numeric value
    # maybe we should scale it? Answer: No, we should not scale the target variable
    #y_train_encoded,y_test_encoded = encode_y_data(y_train,y_test)
    