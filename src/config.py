"""
    Module for the configuraziont of the project.
    It contains all the common variables needed during the piepelines.
"""
from os.path import join
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent

MODEL_FOLDER = join(ROOT_FOLDER, "model")
OUT_FOLDER = join(ROOT_FOLDER, "src", "out")
UTILS_FOLDER = join(ROOT_FOLDER, "src", "utils")
TEST_FOLDER = join(ROOT_FOLDER, "src", "test")

DATASET_FOLDER = join(ROOT_FOLDER, "dataset")
DATASET_PREPROCESSING_FOLDER = join(DATASET_FOLDER, "preprocessing")

DATASET_FOLDER_SMALL = join(DATASET_FOLDER, "small")
DATASET_FOLDER_MEDIUM = join(DATASET_FOLDER, "medium")
DATASET_FOLDER_LARGE = join(DATASET_FOLDER, "large")

SEED_VALUE = int(100)
TEST_SIZE = 0.2

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

LIST_OF_ML_MODELS = [
    "logistic_regression",
    "naive_bayes",
    "knn",
    "random_forest",
    "svm",
    "perceptron"]

LIST_OF_PREPROCESSING = [
    "min_max_scaler",
    "standard_scaler",
    "select_percentile",
    "pca",
    "fast_ica",
    "feature_agglomeration",
    "polynomial_features",
    "radial_basis_function_sampler"
]

# TO DO
LIST_OF_METAFEATURES = []
