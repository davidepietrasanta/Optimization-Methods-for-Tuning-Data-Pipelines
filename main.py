"""
    Main module
"""
from os.path import join
from src.config import DATASET_FOLDER_MEDIUM #, DATASET_FOLDER
#from src.utils.dataset_selection import select_datasets, check_missing_values
from src.utils.metafeatures_extraction import metafeatures_extraction #, metafeature

if __name__ == '__main__':
    VERBOSE = True
    dataset_path = join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")

    #select_datasets(size='medium', verbose=verbose)
    #print( check_missing_values(DATASET_FOLDER_MEDIUM) )
    #metafeature(dataset_path, verbose= verbose)
    metafeatures_extraction( DATASET_FOLDER_MEDIUM, verbose= VERBOSE)
