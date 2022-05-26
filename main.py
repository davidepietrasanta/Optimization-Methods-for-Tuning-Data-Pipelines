import os
from src.config import DATASET_FOLDER
from src.utils.dataset_selection import select_datasets
from src.utils.metafeatures_extraction import metafeature


if __name__ == '__main__':

    #select_datasets(size='medium', verbose=True)
    dataset_path = os.path.join(DATASET_FOLDER, "medium/artificial-characters.csv")
    metafeature(dataset_path)