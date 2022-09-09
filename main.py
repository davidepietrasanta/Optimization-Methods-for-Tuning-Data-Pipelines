"""
    Main module
"""
from os.path import join
import logging
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER # pylint: disable=unused-import
from src.config import METAFEATURES_FOLDER, METAFEATURES_MODEL_FOLDER # pylint: disable=unused-import
from src.utils.metalearner import train_metalearner, hyper_parameters_optimization # pylint: disable=unused-import
from src.utils.data_preparation import data_preparation, delta_or_metafeatures # pylint: disable=unused-import
from src.utils.preprocessing_improvement import predicted_improvement, one_step_bruteforce # pylint: disable=unused-import
from src.utils.preprocessing_improvement import best_one_step_bruteforce, max_in_dict # pylint: disable=unused-import
from src.utils.preprocessing_improvement import pipeline_experiments, preprocessing_experiment # pylint: disable=unused-import
from src.utils.metafeatures_extraction import metafeature # pylint: disable=unused-import
from src.utils.experiments import pipeline_true_experiments, experiment_on_dataset # pylint: disable=unused-import

if __name__ == '__main__':
    VERBOSE = True
    if VERBOSE:
        VERBOSITY_LEVEL = logging.INFO
    else:
        VERBOSITY_LEVEL = logging.CRITICAL

    formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(VERBOSITY_LEVEL)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(pathname)s \n'+
         '%(funcName)s (line:%(lineno)d) - '+
         '%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log", mode='w'),
        stream_handler,
    ]
    )
    logging.info("************START************")

    prova = join(DATASET_FOLDER, 'prova')

    # pylint: disable=pointless-string-statement
    """
    data_preparation(
        data_path=DATASET_FOLDER_MEDIUM, #DATASET_FOLDER_MEDIUM #prova
        data_selection = False,
        data_preprocess = False, #True
        metafeatures_extraction = False, #True
        model_training = False, #True
        quotient=True)

    delta_path = join(METAFEATURES_FOLDER, "delta.csv")

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='random_forest',
        tuning=False)

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='random_forest',
        tuning=True)

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='knn',
        tuning=False)

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='knn',
        tuning=True)

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='gaussian_process',
        tuning=False)

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='gaussian_process',
        tuning=True)

    """
    #emnist-balanced-test.csv #wine-quality-white.csv
    new_dataset = join(
            DATASET_FOLDER,
            join('Test', 'wine-quality-white.csv')
            )

    list_of_experiments = [
        ['standard_scaler'],
        ['standard_scaler', 'pca'],
        ['standard_scaler', 'pca', 'feature_agglomeration'],
        ['standard_scaler', 'feature_agglomeration'],
        ['standard_scaler', 'feature_agglomeration', 'pca'],
        ['pca'],
        ['pca', 'feature_agglomeration'],
        ['feature_agglomeration'],
        ['feature_agglomeration', 'pca'],
    ]

    results = experiment_on_dataset(
        dataset_path = new_dataset,
        algorithm = 'naive_bayes',
        list_of_experiments = list_of_experiments)

    logging.info("results = %s", str(results) )

    logging.info("************END************")

# TO DO:
# Optimization part
# Knn regressor, better with 'uniform' of 'distance' weights?
# Tuning Metalearner model

# PYLINT WARNINGS
# pylint: disable=too-many-arguments
# Consider a dict or an object to store the data and than pass it.
# *args (https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists)
# (https://www.geeksforgeeks.org/args-kwargs-python/)
