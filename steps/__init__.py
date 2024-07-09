# Description: This file is used to import all the functions from the steps folder.

# from feature_engineering folder
from .feature_engineering.update_data import update_data
from .feature_engineering.load_data import load_data
from .feature_engineering.split_data import split_data
from .feature_engineering.create_preprocessing_pipeline import create_preprocessing_pipeline
from .feature_engineering.feature_preprocessor import feature_preprocessor

# from training folder
from .training.hp_tuning import hp_tuning
from .training.model_trainer import model_trainer
from .training.evaluate_model import evaluate_model

# from inference folder
from .inference.inference_data_loader import inference_data_loader
from .inference.inference_preprocessing import inference_preprocessing
from .inference.prediction_service_loader import prediction_service_loader
from .inference.predictor import predictor


## OLD
# from .create_model import create_model
# from .evaluate import evaluate
# from .feature_engineering.load_data import load_data
# from .feature_engineering.update_data import update_data
# from .feature_engineering.split_data import split_data
# from .feature_engineering_OLD import feature_engineering