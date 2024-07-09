from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from functools import partial
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split

def objective(trial, model_class, X_train, y_train):
    """
    Diese Funktion definiert das Ziel für die Optuna-Optimierung.
    """
    params = {}
    
    # 1. Define the hyperparameters to optimize for a Regression problem
    if model_class == RandomForestRegressor:
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['max_depth'] = trial.suggest_int('max_depth', 1, 30)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20)
        model = model_class(**params, random_state=42)
    elif model_class == XGBRegressor:
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
        params['max_depth'] = trial.suggest_int('max_depth', 1, 10)
        params['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        params['subsample'] = trial.suggest_discrete_uniform('subsample', 0.6, 0.95, 0.05)
        params['colsample_bytree'] = trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.95, 0.05)
        model = model_class(**params, random_state=42)
    
    # 2. Split the data into training and validation data
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 3. Train the model
    model.fit(X_train_split, y_train_split)
    
    # 4. Evaluate the model with the rmse
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    return rmse, model

@step
def hp_tuning(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'random_forest', trials: int = 100) -> Annotated[dict, "Best hyperparameters"]:
    """
    Diese Funktion optimiert die Hyperparameter eines Modells mit Optuna.
    """
    # 1. Choose the model class based on the model_type
    if model_type == 'random_forest':
        model_class = RandomForestRegressor
    elif model_type == 'xgboost':
        model_class = XGBRegressor
    else:
        raise ValueError("Unbekannter Modelltyp. Unterstützte Typen: 'random_forest', 'xgboost'")
    
    # 2. Define the objective function
    obj = partial(objective, model_class=model_class, X_train=X_train, y_train=y_train)
    
    # 3. Create a study and optimize it
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=trials)
    
    # 4. Get the best hyperparameters
    best_params = study.best_params
    print(best_params, type(best_params))
    
    return best_params




# from zenml import step
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error
# import optuna
# import pandas as pd
# from functools import partial
# from typing_extensions import Annotated

# def objective(trial,X_train,y_train):
#     """
#         Objective function for the hyperparameter tuning.
#         Here we define the hyperparameters that we want to optimize.
#     Args:
#         trial: Optuna trial object 
#         X_train: Training data
#         y_train: Training labels 
#     """
    
#     # 1. Define the hyperparameters to optimize for a Regression problem
#     max_depth = trial.suggest_int('max_depth', 2, 32)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 32)
#     criterion = trial.suggest_categorical('criterion', ['mse', 'friedman_mse', 'mae'])
    
#     # 2. Create the model with the hyperparameters
#     model = DecisionTreeRegressor(
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         criterion=criterion,
#         random_state=42
#     )
    
#     # 3. Split the data into training and validation data
#     X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     # 4. Train the model
#     model.fit(X_train_split, y_train_split)
    
#     # 5. Evaluate the model
#     y_pred = model.predict(X_val)
#     mse = mean_squared_error(y_val, y_pred)
    
#     return mse

# @step
# def hp_tuning_regression(X_train: pd.DataFrame, y_train: pd.Series, trials: int = 100) -> Annotated[dict, "Best hyperparameters"]:
#     """
#     Dieser Schritt optimiert die Hyperparameter eines Decision Tree Regressionsmodells mit Optuna.
#     """
    
#     # 1. Define the objective function
#     obj = partial(objective, X_train=X_train, y_train=y_train)
    
#     # 2. Create a study
#     study = optuna.create_study(direction="minimize")  # Für Regression wird oft minimize verwendet
    
#     # 3. Optimize the study
#     study.optimize(obj, n_trials=trials)
    
#     # 4. Get the best hyperparameters
#     best_params = study.best_params
    
#     return best_params
    