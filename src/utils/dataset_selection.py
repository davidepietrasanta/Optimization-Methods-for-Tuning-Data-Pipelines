import openml
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from ..config import DATASET_FOLDER
from ..config import DATASET_CONSTRAIN
from ..config import SEED_VALUE
from openml.tasks import TaskType

def select_datasets(size='medium', save_path = DATASET_FOLDER, verbose=False):
    """
        Select a series of databases from OpenML based on the 'size' 
        and save them in the folder 'save_path'.
        Dataset with missing values are not considered.

        :param size: Should be 'small', 'medium' or 'large'.
        It decide the size of the datasets selected.
        :param save_path: Path where the datasets are saved.
        :param verbose: If True more info are printed.
        
        :return: List of names of selected datasets.
    """ 

    size = size.lower()
    if size not in ["small", "medium", "large"]:
        raise ValueError("size should be 'small', 'medium' or 'large'")

    # Get a list of the datasets
    task = openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION)
    task = pd.DataFrame.from_dict(task, orient="index")

    openml_df = openml.tasks.list_tasks(
        task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
    )
    # Just consider 10-fold Crossvalidation
    # TO DO: See if it's ok or useless
    openml_df = openml_df.query('estimation_procedure == "10-fold Crossvalidation"')
    # Drop useless columns
    # ttid tid did target_feature
    openml_df = openml_df.drop(['ttid', 'task_type', 'status', 
                            'estimation_procedure',
                            'source_data', 'cost_matrix'], axis=1)
    # Discard the datasets with missing values
    openml_df = openml_df.loc[openml_df['NumberOfMissingValues'] == 0]

    # Constrain and Shuffle/Sample
    constrain = DATASET_CONSTRAIN[size]
    selected_dataset = openml_df[
        ( openml_df.NumberOfInstances <= constrain["MaxNumberOfInstances"] ) & 
        ( openml_df.NumberOfInstances >= constrain["MinNumberOfInstances"] ) & 
        ( openml_df.NumberOfClasses <= constrain["MaxNumberOfClasses"] ) & 
        ( openml_df.NumberOfClasses >= constrain["MinNumberOfClasses"] ) & 
        ( openml_df.NumberOfFeatures <= constrain["MaxNumberOfFeatures"] ) & 
        ( openml_df.NumberOfFeatures >= constrain["MinNumberOfFeatures"] ) 
        ].sample(frac=1, random_state=SEED_VALUE)

    # If size is medium we can also take the benchmark datasets from OpenML.
    # They are around 100.
    if size == "medium":
        # Benchmark dataset
        tasks_OpenML100 = openml.tasks.list_tasks(tag="OpenML100", output_format="dataframe")
        tasks_OpenMLCC18 = openml.tasks.list_tasks(tag="OpenML-CC18", output_format="dataframe")
        # Concat of datasets
        tasks = pd.concat( [tasks_OpenMLCC18,tasks_OpenML100] )
        # Drop duplicated
        tasks.drop_duplicates(subset ="tid", inplace = True)
        # Discard the datasets with missing values
        tasks = tasks.loc[tasks['NumberOfMissingValues'] == 0]

        selected_dataset = pd.concat( [tasks, selected_dataset] )

        if verbose:
            print("Since the size selected is 'medium' we also select from OpenML benchmark datasets.")
            print("We have found " + str(len(tasks)) + " datasets from the benchmark.")


    # Drop duplicated name and tid
    selected_dataset.drop_duplicates(subset ="name", inplace = True)
    selected_dataset.drop_duplicates(subset ="tid", inplace = True)

    n_dataset = DATASET_CONSTRAIN['n_default_datasets']
    if( len(selected_dataset) < n_dataset ):
        n_dataset = len(selected_dataset)

    if verbose:
        print( str(len(selected_dataset)) + " datasets were found, " + str(n_dataset) + " were selected.")

    list_dataset_name = []
    actual_dataset_num = 0


    for i, (index, dataset) in enumerate(selected_dataset.iterrows()):

        dataset_name = dataset['name']

        try:
            task_openML = openml.tasks.get_task(dataset['tid'])
            dataset_openML = task_openML.get_dataset()
            X, y, categorical_indicator, attribute_names = dataset_openML.get_data(
            target=dataset_openML.default_target_attribute, dataset_format="dataframe"
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

            actual_dataset_num = actual_dataset_num + 1
            list_dataset_name.append( dataset_name )

            if verbose:
                    print( str(actual_dataset_num) + "/" + str(n_dataset) + " " + dataset_name)

        except:
            if verbose:
                print( "Error while dowloading, dataset skipped" )


        if actual_dataset_num >= n_dataset:
            break

    if verbose:
        print(str(len(list_dataset_name)) + " datasets were actually downloaded.")

    return list_dataset_name


def check_missing_values(datasets_path, verbose=False):
    """
        Check if datasets in a folder has missing values.
        If true return all the name of the datasets with missing values.
        Datasets should be csv format files.

        :param datasets_path: Folder where the datasets are checked.
        :param verbose: If True more info are printed.
        
        :return: List of names of datasets with missing values.
    """ 
    missing_values_datasets = []

    list_datasets = [f for f in listdir(datasets_path) if isfile(join(datasets_path, f))]

    for dataset_name in list_datasets:
        if verbose:
            print( "Checking '" + dataset_name + "'..." )

        dataset_path = join(datasets_path, dataset_name)

        df = pd.read_csv (dataset_path)
        count_missing_values = df.isnull().values.any()
        if( count_missing_values > 0 ):
            missing_values_datasets.append(dataset_path)

            if verbose:
                print( "'" + dataset_name + "' with missing values." )

    return missing_values_datasets
    