"""
    Main module
"""
from os.path import join # pylint: disable=unused-import
import pandas as pd # pylint: disable=unused-import
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER # pylint: disable=unused-import
from src.config import METAFEATURES_FOLDER, list_of_metafeatures # pylint: disable=unused-import
from src.config import LIST_OF_ML_MODELS, LIST_OF_PREPROCESSING # pylint: disable=unused-import
from src.utils.dataset_selection import select_datasets # pylint: disable=unused-import
from src.utils.metafeatures_extraction import metafeatures_extraction # pylint: disable=unused-import
from src.utils.machine_learning_algorithms import machine_learning_algorithm # pylint: disable=unused-import
from src.utils.preprocessing_methods import preprocess_dataset, categorical_string_to_number # pylint: disable=unused-import
from src.utils.preprocessing_methods import preprocess_all_datasets # pylint: disable=unused-import

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")
    dataset_path_2 = join(DATASET_FOLDER_MEDIUM, "analcatdata_dmft.csv")

    # Takes a long time
    #select_datasets(size='medium', verbose=verbose)

    #for algorithm in LIST_OF_ML_MODELS:
    #    machine_learning_algorithm(dataset_path=dataset_path, algorithm=algorithm, verbose=True)

    #for method in LIST_OF_PREPROCESSING:
    #    x = preprocess_dataset(dataset_path=dataset_path, method=method, verbose=True)

    prova = join(DATASET_FOLDER, 'prova')
    #preprocess_all_datasets(
    #    datasets_path = DATASET_FOLDER_MEDIUM,
    #    save_path = DATASET_FOLDER_MEDIUM,
    #    verbose = True)

    metafeatures_extraction(
        datasets_path = prova,
        save_path = METAFEATURES_FOLDER,
        verbose = True)

    print( list_of_metafeatures() )
