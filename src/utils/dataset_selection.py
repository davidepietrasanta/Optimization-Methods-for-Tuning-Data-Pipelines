import openml
import os
import pandas as pd
from openml.datasets import get_dataset
from ..config import DATASET_FOLDER
from ..config import DATASET_CONSTRAIN

def select_datasets(size='small', save_path = DATASET_FOLDER, verbose=False):
    """
        Select a series of databases from OpenML based on the 'size' 
        and save them in the folder 'save_path'.

        :param size: Should be 'small', 'medium' or 'large'.
        It decide the size of the datasets selected.
        :param save_path: Path where the datasets are saved.
        
        :return: List of names of selected datasets.
    """ 

    size = size.lower()
    if size not in ["small", "medium", "large"]:
        raise ValueError("size should be 'small', 'medium' or 'large'")

    # Get a list of the datasets
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # Discard the datasets with missing values
    openml_df = openml_df.loc[openml_df['NumberOfMissingValues'] == 0]
    
    constrain = DATASET_CONSTRAIN[size]

    ascending_value = False
    if size == 'small':
        ascending_value = True


    selected_dataset = openml_df[
        ( openml_df.NumberOfInstances <= constrain["MaxNumberOfInstances"] ) & 
        ( openml_df.NumberOfInstances >= constrain["MinNumberOfInstances"] ) & 
        ( openml_df.NumberOfClasses <= constrain["MaxNumberOfClasses"] ) & 
        ( openml_df.NumberOfClasses >= constrain["MinNumberOfClasses"] ) & 
        ( openml_df.NumberOfFeatures <= constrain["MaxNumberOfFeatures"] ) & 
        ( openml_df.NumberOfFeatures >= constrain["MinNumberOfFeatures"] ) 
        ].sort_values(["NumberOfInstances"], ascending=ascending_value)

    n_dataset = DATASET_CONSTRAIN['n_default_datasets']
    if( len(selected_dataset) < n_dataset ):
        n_dataset = len(selected_dataset)

    if verbose:
        print( str(len(selected_dataset)) + " datasets were found, " + str(n_dataset) + " were selected.")

    list_dataset_name = []
    actual_dataset_num = 0


    for i, (index, dataset) in enumerate(selected_dataset.iterrows()):

        dataset_id = index
        dataset_name = dataset['name']
        if dataset_name not in list_dataset_name:
            actual_dataset_num = actual_dataset_num + 1
            list_dataset_name.append( dataset_name )
            ds = openml.datasets.get_dataset(dataset_id)

            X, y, categorical_indicator, attribute_names = ds.get_data(
            target=ds.default_target_attribute, dataset_format="dataframe"
            )
            # Add y (target) label to the dataframe
            X['y'] = y

            # Save the dataframe
            path = os.path.join(save_path, size)
            # Create the dir if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            path = os.path.join(path, dataset_name + '.csv')

            X.to_csv(path, index=False)

            if verbose:
                    print( str(actual_dataset_num) + "/" + str(n_dataset) + " " + dataset_name)
        else:
            if verbose:
                print( "(duplicate) " + dataset_name)


        if actual_dataset_num >= n_dataset:
            break

    if verbose:
        print(str(len(list_dataset_name)) + " datasets were actually downloaded.")

    return list_dataset_name


