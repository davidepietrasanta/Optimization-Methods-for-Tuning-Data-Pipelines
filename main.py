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

    #path = join(prova,'min_max_scaler', 'artificial-characters.csv')
    #metafeature( path, verbose=True)

    #data_preparation(
    #    data_path=prova,
    #    data_selection = False,
    #    data_preprocess = True,
    #    metafeatures_extraction = True,
    #    model_training = True,
    #    verbose=VERBOSE)

# Check Return None
# Make private all function not used outside file (for all the files)
