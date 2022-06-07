"""
    Main module
"""
from os.path import join
import pandas as pd
from src.config import DATASET_FOLDER_MEDIUM ,DATASET_FOLDER
from src.config import LIST_OF_ML_MODELS, LIST_OF_PREPROCESSING
#from src.utils.dataset_selection import select_datasets
#from src.utils.metafeatures_extraction import metafeatures_extraction
from src.utils.machine_learning_algorithms import machine_learning_algorithm
from src.utils.preprocessing_methods import preprocess_dataset, categorical_string_to_number
from src.utils.preprocessing_methods import preprocess_all_datasets

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
    preprocess_all_datasets(datasets_path=DATASET_FOLDER_MEDIUM,save_path= DATASET_FOLDER_MEDIUM,verbose=True)

    #dataset = pd.read_csv(dataset_path_2)
    #print( dataset.head() )
    #print( categorical_string_to_number(dataset).head() )