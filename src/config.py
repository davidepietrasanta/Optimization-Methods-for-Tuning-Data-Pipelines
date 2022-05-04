import os
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent

CODE_FOLDER = os.path.join(ROOT_FOLDER, "src", "code")
DATASET_FOLDER = os.path.join(ROOT_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")
OUT_FOLDER = os.path.join(ROOT_FOLDER, "src", "out")
UTILS_FOLDER = os.path.join(ROOT_FOLDER, "src", "utils")

DATASET_CONSTRAIN = {
    'n_default_datasets': 200,
    'small': {
        'MinNumberOfInstances': 0 ,
        'MaxNumberOfInstances': 1000 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 5 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 50,
    },
    'medium': {
        'MinNumberOfInstances': 1000 ,
        'MaxNumberOfInstances': 1000000,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 25 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 200,
    },
    'large': {
        'MinNumberOfInstances': 1000000 ,
        'MaxNumberOfInstances': 1000000000 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 50 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 500,
    }

} 


LIST_OF_PREPROCESSING = []

LIST_OF_MODELS = []
