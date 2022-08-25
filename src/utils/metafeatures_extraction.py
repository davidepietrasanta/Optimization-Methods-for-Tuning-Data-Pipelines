"""
    Module for the extraction of the metafeatures.
"""
import math
import os
from os import listdir
from os.path import isfile, join, basename
import logging
import pandas as pd
import numpy as np
import skdim
from pymfe.mfe import MFE
from src.config import METAFEATURES_FOLDER
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
    datasets_metafeatures = []

    missing_values_datasets = check_missing_values(datasets_path)

    for i, dataset_name in enumerate(list_datasets):
        logging.info(
            "Extracting metafeatures from: %s'...(%s/%s) [%s]",
            dataset_name,str(i+1), str(n_datasets), str(basename(datasets_path))
            )

        dataset_path = join(datasets_path, dataset_name)
        if dataset_path not in missing_values_datasets:
            metafeatures_data = metafeature(dataset_path)
            if metafeatures_data is not None:
                metafeatures_data['dataset_name'] = dataset_name
                datasets_metafeatures.append( metafeatures_data )
        else:
            logging.info("'%s' skipped because of missing values.", dataset_name)


    dataset = pd.DataFrame(datasets_metafeatures)

    if name_saved_csv is None:
        name_saved_csv = "metafeatures_data.csv"

    dataset.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset.to_csv(join(save_path, name_saved_csv), index=True)

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
        if 'y' in list(dataset.columns):
            y_label = dataset["y"].to_list()
            x_label = dataset.drop(["y"], axis=1).to_numpy()
        else:
            # This is because we may want to work with pre-processed data
            # Preprocess can change che nature of the features so it's not
            # possible to keep the original features name.
            y_label = dataset.iloc[: , -1].tolist()
            x_label = dataset.iloc[: , :-1].to_numpy()

        # Extract general, statistical and information-theoretic measures
        mfe = MFE(groups=["general", "statistical", "info-theory"], suppress_warnings= not verbose)
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
 