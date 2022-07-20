"""
    Main module
"""
from os.path import join # pylint: disable=unused-import
import pandas as pd # pylint: disable=unused-import
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER # pylint: disable=unused-import
from src.config import METAFEATURES_FOLDER, list_of_metafeatures # pylint: disable=unused-import
from src.utils.metalearner import data_preparation, train_metalearner # pylint: disable=unused-import
from src.utils.metalearner import choose_performance_from_metafeatures, split_train_test # pylint: disable=unused-import
from src.utils.metalearner import delta_or_metafeatures # pylint: disable=unused-import

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")
    dataset_path_2 = join(DATASET_FOLDER_MEDIUM, "analcatdata_dmft.csv")
    prova = join(DATASET_FOLDER, 'prova')

    path = join(prova,'min_max_scaler', 'artificial-characters.csv')

    data_preparation(
        data_path=prova,
        data_selection = False,
        data_preprocess = False,
        metafeatures_extraction = False,
        model_training = False,
        quotient=False,
        verbose=VERBOSE)

    delta_path = join(METAFEATURES_FOLDER, "delta.csv")

    train_metalearner(
        metafeatures_path = delta_path,
        algorithm='random_forest',
        verbose=VERBOSE)

# TO DO:
# Make private all function not used outside file (for all the files)
# Optimization part / Robust Optimization
# MLflow

# PYLINT WARNINGS
# # pylint: disable=too-many-arguments
# Consider a dict or an object to store the data and than pass it.
# *args (https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists)
# (https://www.geeksforgeeks.org/args-kwargs-python/)
