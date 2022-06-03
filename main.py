"""
    Main module
"""
from os.path import join
from src.config import DATASET_FOLDER_MEDIUM #, DATASET_FOLDER
#from src.utils.dataset_selection import select_datasets, check_missing_values
#from src.utils.metafeatures_extraction import metafeatures_extraction , metafeature
from src.utils.machine_learning_algorithms import machine_learning_algorithm

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")

    #select_datasets(size='medium', verbose=verbose)
    #print( check_missing_values(DATASET_FOLDER_MEDIUM) )
    #metafeature(dataset_path, verbose= verbose)
    #metafeatures_extraction( DATASET_FOLDER_MEDIUM, verbose= VERBOSE)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='logistic_regression', verbose=True)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='naive_bayes', verbose=True)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='knn', verbose=True)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='random_forest', verbose=True)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='svm', verbose=True)
    machine_learning_algorithm(dataset_path=dataset_path, algorithm='perceptron', verbose=True)
