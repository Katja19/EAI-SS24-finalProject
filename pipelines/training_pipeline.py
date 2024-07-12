from steps import hp_tuning,model_trainer,evaluate_model
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.materializers.pandas_materializer import PandasMaterializer

""" 
This pipeline uses the data from the Feature Engineering pipeline. 
We can get this data using our ZenML client, as we did before, to extract the mae value.
It then creates a decision tree and deploys it using MLflow when a deployment decision is made.
"""
@pipeline(enable_cache=False)
def training_pipeline(model_variant:str, model_type:str):
    """ 
        Pipeline to train and deploy a machine learning model using preprocessed and encoded datasets.
    """
    
    # 1. Get the preprocessed X data and the non-encoded y data from the Feature Engineering pipeline
    client = Client()
    X_train = client.get_artifact_version("X_train_preprocessed")
    X_test = client.get_artifact_version("X_test_preprocessed")
    y_train = client.get_artifact_version("y_train") # not encoded cause it is a regression problem
    y_test = client.get_artifact_version("y_test") # not encoded cause it is a regression problem
    
    # # transform the artifacts to pandas DataFrames respectively Series
    # X_train = X_train_artifact.read(materializer_class=PandasMaterializer)
    # print(type(X_train))
    # X_test = X_test_artifact.load()
    # print(type(X_test))
    # y_train = y_train_artifact.load().to_pandas()
    # #print(type(y_train))
    # y_test = y_test_artifact.load().to_pandas()
    # #print(type(y_test))

    # 2. Get the best hyperparameters for the model
    best_parameters = hp_tuning(X_train,y_train,model_type, trials=100)
    
    # 3. Train the model and get the in-sample score (RMSE)
    model,in_sample_rmse =model_trainer(X_train,y_train,model_type,best_parameters)
    
    # 4. Evaluate the model using the test data, here we calculate and save the out-of-sample score (MSE) and other metrics
    deploy = evaluate_model(model,X_test,y_test)
    
    # 5. Deploy the model using MLflow if the deployment decision deploy == True
    mlflow_model_deployer_step(model=model,deploy_decision=deploy,workers=1) 