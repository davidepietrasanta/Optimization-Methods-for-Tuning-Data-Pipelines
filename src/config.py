import os
from pathlib import Path

from numpy import integer

ROOT_FOLDER = Path(__file__).parent.parent

CODE_FOLDER = os.path.join(ROOT_FOLDER, "src", "code")
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")
OUT_FOLDER = os.path.join(ROOT_FOLDER, "src", "out")
UTILS_FOLDER = os.path.join(ROOT_FOLDER, "src", "utils")

DATASET_FOLDER = os.path.join(ROOT_FOLDER, "dataset")

DATASET_FOLDER_SMALL = os.path.join(DATASET_FOLDER, "small")
DATASET_FOLDER_MEDIUM = os.path.join(DATASET_FOLDER, "medium")
DATASET_FOLDER_LARGE = os.path.join(DATASET_FOLDER, "large")

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

# TO DO
LIST_OF_METAFEATURES = []

LIST_OF_PREPROCESSING = []

LIST_OF_MODELS = []
 