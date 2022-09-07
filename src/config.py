"""
    Module for the configuration of the project.
    It contains all the common variables needed during the piepelines.
"""
import shutil
from os.path import join, exists
from pathlib import Path
from skopt.space import Categorical, Integer, Real
from pandas import read_csv

ROOT_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = join(ROOT_FOLDER, "data")

MODEL_FOLDER = join(DATA_FOLDER, "model")
METAFEATURES_MODEL_FOLDER = join(MODEL_FOLDER, "metalearner")
TEMP_MODEL_FOLDER = join(MODEL_FOLDER, "temp")
METAFEATURES_FOLDER = join(DATA_FOLDER, "metafeatures")

UTILS_FOLDER = join(ROOT_FOLDER, "src", "utils")
TEST_FOLDER = join(ROOT_FOLDER, "src", "test")

DATASET_FOLDER = join(DATA_FOLDER, "dataset")
DATASET_PREPROCESSING_FOLDER = join(DATASET_FOLDER, "preprocessing")

DATASET_FOLDER_SMALL = join(DATASET_FOLDER, "small")
DATASET_FOLDER_MEDIUM = join(DATASET_FOLDER, "medium")
DATASET_FOLDER_LARGE = join(DATASET_FOLDER, "large")

SEED_VALUE = int(100)
TEST_SIZE = 0.20

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
        'MinNumberOfFeatures': 10,
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

CATEGORICAL_LIST_OF_ML_MODELS = [
    1, #logistic_regression
    2, #naive_bayes
    3, #knn
    4, #random_forest
    5, #svm
    6 #perceptron
    ]

LIST_OF_ML_MODELS_FOR_METALEARNING = [
    "knn",
    "random_forest",
    "gaussian_process"]

SEARCH_SPACE = {}

SEARCH_SPACE['knn'] = {
    'p' : Integer(1, 5),
    'leaf_size' : Integer(5,100),
    'weights' : Categorical(['uniform', 'distance']),
    'algorithm' : Categorical(['auto', 'ball_tree', 'kd_tree'])
}

SEARCH_SPACE['random_forest'] = {
    'n_estimators' : Integer(50, 200),
    'criterion' : Categorical(['squared_error','absolute_error','poisson'])
}

SEARCH_SPACE['gaussian_process'] = {
    'n_restarts_optimizer' : Integer(0, 10),
    'alpha' : Real(1e-12, 1e-3, 'log-uniform')
}

LIST_OF_PREPROCESSING = [
    "min_max_scaler",
    "standard_scaler",
    "select_percentile",
    "pca",
    "fast_ica",
    "feature_agglomeration",
    "radial_basis_function_sampler"
]

def list_of_metafeatures(metafeatures_path:None or str=None) -> list:
    """
       Used to know the list of the metafeatures used during the pipeline.

        :param metafeatures_path: Path where the metafeatures are saved.
         Datasets should be in a CSV format.

        :return: A list of metafeatures.
    """
    if metafeatures_path is None:
        metafeatures_path = join(METAFEATURES_FOLDER, 'delta.csv')

    metafeatures_csv = read_csv(metafeatures_path)
    list_of_columns = list(metafeatures_csv.columns)
    drop_list = ["dataset_name", "algorithm", "preprocessing", "performance"]
    cleaned_list = [ x for x in list_of_columns if x not in drop_list ]

    return cleaned_list

def delete_dir(dir_path:str) -> bool:
    """
        Delete the 'dir_path' directory and all its files
    """
    if exists(dir_path):
        shutil.rmtree(dir_path)

    assert not exists(dir_path)
    return True
