"""
    Main module
"""
from os.path import join
from src.config import DATASET_FOLDER_MEDIUM #,DATASET_FOLDER
from src.config import LIST_OF_ML_MODELS, LIST_OF_PREPROCESSING 
#from src.utils.dataset_selection import select_datasets
#from src.utils.metafeatures_extraction import metafeatures_extraction
#from src.utils.machine_learning_algorithms import machine_learning_algorithm
from src.utils.preprocessing_methods import preprocess_dataset

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")

    #select_datasets(size='medium', verbose=verbose)

    #for algorithm in LIST_OF_ML_MODELS:
    #    machine_learning_algorithm(dataset_path=dataset_path, algorithm=algorithm, verbose=True)

    for method in LIST_OF_PREPROCESSING:
        x = preprocess_dataset(dataset_path=dataset_path, method=method, verbose=True)
