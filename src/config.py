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
        'MinNumberOfInstances': 0 ,
        'MaxNumberOfInstances': 1000 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 5 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 50,
    },
    'large': {
        'MinNumberOfInstances': 0 ,
        'MaxNumberOfInstances': 1000 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 5 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 50,
    }

} 


LIST_OF_PREPROCESSING = []

LIST_OF_MODELS = []
