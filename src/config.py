import os
from pathlib import Path

from numpy import integer

ROOT_FOLDER = Path(__file__).parent.parent

CODE_FOLDER = os.path.join(ROOT_FOLDER, "src", "code")
DATASET_FOLDER = os.path.join(ROOT_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")
OUT_FOLDER = os.path.join(ROOT_FOLDER, "src", "out")
UTILS_FOLDER = os.path.join(ROOT_FOLDER, "src", "utils")

SEED_VALUE = int(100)

DATASET_CONSTRAIN = {
    'n_default_datasets': 200,
    'small': {
        'MinNumberOfInstances': 0 ,
        'MaxNumberOfInstances': 500 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 5 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 1000,
    },
    'medium': {
        'MinNumberOfInstances': 500 ,
        'MaxNumberOfInstances': 100000,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 25 ,
        'MinNumberOfFeatures': 10,
        'MaxNumberOfFeatures': 5000,
    },
    'large': {
        'MinNumberOfInstances': 100000 ,
        'MaxNumberOfInstances': 1000000000 ,
        'MinNumberOfClasses': 0 ,
        'MaxNumberOfClasses': 50 ,
        'MinNumberOfFeatures': 0,
        'MaxNumberOfFeatures': 5000,
    }

} 


LIST_OF_PREPROCESSING = []

LIST_OF_MODELS = []
