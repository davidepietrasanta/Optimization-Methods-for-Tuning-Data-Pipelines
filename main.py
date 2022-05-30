import os
from src.config import DATASET_FOLDER, DATASET_FOLDER_MEDIUM
from src.utils.dataset_selection import select_datasets, check_missing_values
from src.utils.metafeatures_extraction import metafeature, metafeatures_extraction


if __name__ == '__main__':
    verbose = True
    dataset_path = os.path.join(DATASET_FOLDER_MEDIUM, "artificial-characters.csv")

    #select_datasets(size='medium', verbose=verbose)
    #print( check_missing_values(DATASET_FOLDER_MEDIUM) )
    #metafeature(dataset_path, verbose= verbose)
    metafeatures_extraction( DATASET_FOLDER_MEDIUM, verbose= verbose)

 