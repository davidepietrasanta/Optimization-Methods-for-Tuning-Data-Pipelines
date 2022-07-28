"""
    Module for the selection and download of the dataset to train the meta-learning model.
"""

from os import listdir, makedirs
from os.path import isfile, join, exists
import logging
import pandas as pd
import openml
from openml.tasks import TaskType

from src.config import DATASET_FOLDER
from src.config import DATASET_CONSTRAIN
from src.config import SEED_VALUE
from src.exceptions import exception_logging

def select_datasets(
    size:str ='medium',
    save_path:str = DATASET_FOLDER) -> list:
    """
        Select a series of databases from OpenML based on the 'size'
         and save them in the folder 'save_path'.
         Dataset with missing values are not considered.

        :param size: Should be 'small', 'medium' or 'large'.
        It decide the size of the datasets selected.
        :param save_path: Path where the datasets are saved.

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
        tasks_openml_100 = openml.tasks.list_tasks(tag="OpenML100", output_format="dataframe")
        tasks_openml_cc18 = openml.tasks.list_tasks(tag="OpenML-CC18", output_format="dataframe")
        # Concat of datasets
        tasks = pd.concat( [tasks_openml_cc18,tasks_openml_100] )
        # Drop duplicated
        tasks.drop_duplicates(subset ="tid", inplace = True)
        # Discard the datasets with missing values
        tasks = tasks.loc[tasks['NumberOfMissingValues'] == 0]

        selected_dataset = pd.concat( [tasks, selected_dataset] )

        msg = "Since the size selected is 'medium' we also select from OpenML benchmark datasets."
        logging.info(msg)
        logging.info("We have found %s datasets from the benchmark.",str(len(tasks)))


    # Drop duplicated name and tid
    selected_dataset.drop_duplicates(subset ="name", inplace = True)
    selected_dataset.drop_duplicates(subset ="tid", inplace = True)

    n_dataset = DATASET_CONSTRAIN['n_default_datasets']
    if len(selected_dataset) < n_dataset:
        n_dataset = len(selected_dataset)

    logging.info( "%s datasets were found, %s were selected.",
        str(len(selected_dataset)), str(n_dataset))

    list_dataset_name = []
    actual_dataset_num = 0


    for _, dataset in selected_dataset.iterrows():

        dataset_name = dataset['name']

        try:
            task_openml = openml.tasks.get_task(dataset['tid'])
            dataset_openml = task_openml.get_dataset()
            x_label, y_label, _, _ = dataset_openml.get_data(
            target=dataset_openml.default_target_attribute, dataset_format="dataframe"
            )

            # Add y (target) label to the dataframe
            x_label['y'] = y_label

            # Save the dataframe
            path = join(save_path, size)
            # Create the dir if it doesn't exist
            if not exists(path):
                makedirs(path)

            path = join(path, dataset_name + '.csv')

            x_label.to_csv(path, index=False)

            actual_dataset_num = actual_dataset_num + 1
            list_dataset_name.append( dataset_name )

            logging.info( "%s/%s %s", str(actual_dataset_num), str(n_dataset), dataset_name)

        except Exception: # pylint: disable=broad-except
            msg = "Error while dowloading, dataset skipped"
            exception_logging(msg)


        if actual_dataset_num >= n_dataset:
            break

    logging.info("%s datasets were actually downloaded.",
    str(len(list_dataset_name)))

    return list_dataset_name


def check_missing_values(datasets_path:str) -> list:
    """
        Check if datasets in a folder has missing values.
        If true return all the name of the datasets with missing values.
        Datasets should be csv format files.

        :param datasets_path: Folder where the datasets are checked.

        :return: List of names of datasets with missing values.
    """

    missing_values_datasets = []

    list_datasets = [f for f in listdir(datasets_path) if isfile(join(datasets_path, f))]

    for dataset_name in list_datasets:
        logging.debug("Checking ' %s ...", dataset_name)

        dataset_path = join(datasets_path, dataset_name)

        dataset = pd.read_csv (dataset_path)
        count_missing_values = dataset.isnull().values.any()
        if count_missing_values > 0 :
            missing_values_datasets.append(dataset_path)

            logging.debug("'%s' with missing values.", dataset_name)

    return missing_values_datasets
    