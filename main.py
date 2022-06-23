"""
    Main module
"""
from os.path import join # pylint: disable=unused-import
import pandas as pd # pylint: disable=unused-import
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER # pylint: disable=unused-import
from src.config import METAFEATURES_FOLDER, list_of_metafeatures # pylint: disable=unused-import
from src.config import LIST_OF_ML_MODELS, LIST_OF_PREPROCESSING # pylint: disable=unused-import
from src.utils.metalearner import data_preparation # pylint: disable=unused-import

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")
    dataset_path_2 = join(DATASET_FOLDER_MEDIUM, "analcatdata_dmft.csv")
    prova = join(DATASET_FOLDER, 'prova')

    #path = join(prova,'min_max_scaler', 'artificial-characters.csv')
    #metafeature( path, verbose=True)

    data_preparation(
        data_path=prova,
        data_preprocess = False,
        metafeatures_extraction = False,
        model_training = False,
        verbose=VERBOSE)

    # TO DO:
    # Tutorial
