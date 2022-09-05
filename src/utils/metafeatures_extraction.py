"""
    Module for the extraction of the metafeatures.
"""
import math
import os
from os import listdir
from os.path import isfile, join, basename, exists
import logging
import pandas as pd
import numpy as np
import skdim
from pymfe.mfe import MFE
from src.config import METAFEATURES_FOLDER, SEED_VALUE
from src.exceptions import exception_logging
from .dataset_selection import check_missing_values
from .preprocessing_methods import categorical_string_to_number


def metafeatures_extraction_data(
    datasets_path:str,
    save_path:str = METAFEATURES_FOLDER,
    name_saved_csv:None or str = None) -> pd.DataFrame:
    """
        Given a path to datasets, it return, and save, the matefeatures for each dataset.

        :param datasets_path: Path where the datasets are. Datasets should be in a CSV format.
        :param save_path: Path where the metafeatures are saved.
        :param name_saved_csv: The name of the csv file with all the metafeatures.
         If None the name is 'metafeatures.csv'.

        :return: Metafeatures, as pandas Dataframe.
    """
    list_datasets = sorted([f for f in listdir(datasets_path) if isfile(join(datasets_path, f))])
    n_datasets = len(list_datasets)

    if name_saved_csv is None:
        name_saved_csv = "metafeatures_data.csv"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metafeatures_csv_path = join(save_path, name_saved_csv)

    logging.info("Checking for missing values in the datasets")
    missing_values_datasets = check_missing_values(datasets_path)
    logging.info("Missing values checked")

    for i, dataset_name in enumerate(list_datasets):
        logging.info(
            "Extracting metafeatures from: %s'...(%s/%s) [%s]",
            dataset_name,str(i+1), str(n_datasets), str(basename(datasets_path))
            )

        dataset_path = join(datasets_path, dataset_name)
        if dataset_path not in missing_values_datasets:

            # If the metafeatures are already been extracted we don't do it again
            if not _dataset_in_csv(metafeatures_csv_path, dataset_name):
                metafeatures_data = metafeature(dataset_path)
                if metafeatures_data is not None:
                    metafeatures_data['dataset_name'] = dataset_name

                    metafeatures_df = pd.DataFrame.from_dict([metafeatures_data])

                    # If the file already exist it append the information
                    if exists(metafeatures_csv_path):
                        dataset_df = pd.read_csv(metafeatures_csv_path)
                        metafeatures_df = dataset_df.merge(metafeatures_df, how='outer')

                    metafeatures_df.to_csv(
                        metafeatures_csv_path,
                        index=False)

                    logging.info("'%s' metafeatures saved in csv.", dataset_name)
            else:
                logging.info("'%s' skipped because of already in csv.", dataset_name)

        else:
            logging.info("'%s' skipped because of missing values.", dataset_name)

    dataset = None
    if exists(metafeatures_csv_path):
        dataset = pd.read_csv(metafeatures_csv_path)

        dataset.dropna(
            axis=1,
            how='any',
            thresh=None,
            subset=None,
            inplace=True)

    return dataset


def metafeature(dataset_path:str, verbose:bool =False) -> None or dict:
    """
        Given a path to a single dataset it return the metafeatures of the dataset.

        :param dataset_path: Path where the dataset is.
         It should be in a CSV format.
        :param verbose: If True more info are showed.

        :return: Metafeatures, as Dict, or None.
    """
    try:
        # Read the CSV
        dataset = pd.read_csv(dataset_path)
        dataset = categorical_string_to_number(dataset)

        # Separate X from y
        if "y" in list(dataset.columns):
            y_label = dataset["y"].to_list()
            x_label = dataset.drop(["y"], axis=1).to_numpy()
        else:
            # This is because we may want to work with pre-processed data
            # Preprocess can change che nature of the features so it's not
            # possible to keep the original features name.
            y_label = dataset.iloc[: , -1].tolist()
            x_label = dataset.iloc[: , :-1].to_numpy()

        # Extract general, statistical and information-theoretic measures
        mfe = MFE(
            groups=["general", "statistical", "info-theory"],
            random_state = SEED_VALUE,
            suppress_warnings= not verbose)
        mfe.fit(x_label, y_label)
        features = mfe.extract(suppress_warnings= not verbose )

        logging.debug("Extracted metafeatures with MFE: %s",str(basename(dataset_path)))

        keys = features[0]
        values = features[1]

        dict_ft = {}
        for i, key_i in enumerate(keys):
            # Not insert NaN values
            if not math.isnan(values[i]):
                dict_ft[key_i] = values[i]

        dict_ft['intrinsic_dim'] = intrinsic_dimensionality(x_label)

        logging.debug("Extracted intrinsic dimensionality: %s",str(basename(dataset_path)))

        logging.debug("%s meta-features were extracted from the dataset: %s",
            str(len(dict_ft)), str(dataset_path))

        return dict_ft

    except Exception: # pylint: disable=broad-except
        msg = "Error while extracting metafeature of '" + str(dataset_path) + "', skipped."
        exception_logging(msg)

        return None


def intrinsic_dimensionality(data : np.ndarray) -> int:
    """
        Calculate intrinsic dimensionality.\n
        To know more visit: https://scikit-dimension.readthedocs.io/en/latest/api.html

        :param data: The data in a numpy format.

        :return: The intrinsic dimensionality.

    """
    # Estimate intrinsic dimension
    intrinsic_dim = skdim.id.KNN().fit(data)
    # Get estimated intrinsic dimension
    intrinsic_dim = intrinsic_dim.dimension_

    return intrinsic_dim

def _dataset_in_csv(csv_path:str, name:str) -> bool:
    """
        Check if 'name' already in the csv column 'dataset_name'.

        :param csv_path: Path of the csv file.
        :param name: Name of the dataset you are looking for.

        :return: True if 'name' already in the csv.
    """
    # Open the csv
    if not exists(csv_path):
        return False
    csv_data = pd.read_csv(csv_path)
    datasets_list = csv_data['dataset_name'].values
    # Check if dataset_name is in the column
    return name in datasets_list
