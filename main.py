"""
    Main module
"""
from os.path import join
import logging
from src.config import DATASET_FOLDER
from src.utils.experiments import experiment_on_dataset

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

    #emnist-balanced.csv
    #wine-quality-white.csv

    new_dataset = join(
            DATASET_FOLDER,
            join('Test', 'emnist-balanced.csv')
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
# Gaussian Process prediction with std
# predict(X, return_std=True)
