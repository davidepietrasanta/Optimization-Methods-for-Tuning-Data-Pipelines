"""
    Main module
"""
from os.path import join
import logging
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER # pylint: disable=unused-import
from src.config import METAFEATURES_FOLDER, METAFEATURES_MODEL_FOLDER # pylint: disable=unused-import
from src.utils.metalearner import train_metalearner # pylint: disable=unused-import
from src.utils.data_preparation import data_preparation, delta_or_metafeatures # pylint: disable=unused-import
from src.utils.preprocessing_improvement import predicted_improvement # pylint: disable=unused-import

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
        data_preprocess = True, #True
        metafeatures_extraction = True, #True
        model_training = True,
        quotient=True)
    """

    delta_path = join(METAFEATURES_FOLDER, "delta.csv")

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='gaussian_process')

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='random_forest')

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='knn')

    #emnist-balanced-test.csv #wine-quality-white.csv
    new_dataset = join(
            DATASET_FOLDER,
            join('Test', 'wine-quality-white.csv')
            )

    predicted_improvement(
        dataset_path= new_dataset,
        preprocessing = 'pca',
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_gaussian_process.joblib')
    )

    predicted_improvement(
        dataset_path= new_dataset,
        preprocessing = 'pca',
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    )

    predicted_improvement(
        dataset_path= new_dataset,
        preprocessing = 'pca',
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_knn.joblib')
    )

    logging.info("************END************")

# TO DO:
# Optimization part
# Knn regressor, better with 'uniform' of 'distance' weights?
# Tuning Metalearner model
# 1 Layer all possibilities
# 3 Layer + 3 preprocessing (SS, PCA, FA), with presence and absence

# PYLINT WARNINGS
# pylint: disable=too-many-arguments
# Consider a dict or an object to store the data and than pass it.
# *args (https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists)
# (https://www.geeksforgeeks.org/args-kwargs-python/)
