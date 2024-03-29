"""
    Module for the preprocessing methods.
    It implement the following methods:\n
        - Min-Max Scaler (MMS)\n
        - Standard Scaler (SS)\n
        - Select Percentile (SP)\n
        - Principal Component Analysis (PCA)\n
        - Fast Independent Component Analysis (ICA)\n
        - Feature Agglomeration (FA)\n
        - Polynomial Features (PF)\n
        - Radial Basis Function Sampler (RBFS)\n
"""
import os
from os import listdir
from os.path import isfile, join, exists
import logging
import pandas as pd
from pandas import read_csv
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import cluster
from sklearn.kernel_approximation import RBFSampler
from src.config import SEED_VALUE, DATASET_PREPROCESSING_FOLDER
from src.config import LIST_OF_PREPROCESSING
from src.config import CATEGORICAL_LIST_OF_ML_MODELS, LIST_OF_ML_MODELS
from src.exceptions import CustomValueError, exception_logging

def preprocess_all_datasets(
    datasets_path:str,
    save_path:str = DATASET_PREPROCESSING_FOLDER) -> list:
    """
        Given a path to different datasets, it return, and save, the transformed datasets.

        :param datasets_path: Path where the datasets are. Datasets should be in a CSV format.
       :param method: Preprocessing method selected.
         It should be in
         ["min_max_scaler",
         "standard_scaler",
         "select_percentile",
         "pca",
         "fast_ica",
         "feature_agglomeration",
         "radial_basis_function_sampler"].

        :return: List of sucessfully preprocessed datasets.
    """
    list_datasets = sorted([f for f in listdir(datasets_path) if isfile(join(datasets_path, f))])
    n_datasets = len(list_datasets)
    preprocessed_datasets = []

    for i, dataset_name in enumerate(list_datasets):
        logging.info("Transforming dataset: '%s'...(%d/%d)",
        dataset_name, i+1, n_datasets)

        dataset_path = join(datasets_path, dataset_name)
        n_methods = 0
        for method in LIST_OF_PREPROCESSING:
            try:
                preprocess_dataset(dataset_path, method, save_path)
                n_methods = n_methods + 1
            except Exception:  # pylint: disable=broad-except
                msg = "Error, '" + method + "' skipped in '" + dataset_name + "'."
                exception_logging(msg)

        if n_methods == len(LIST_OF_PREPROCESSING):
            preprocessed_datasets.append(dataset_name)

    return preprocessed_datasets

def preprocess_dataset(
    dataset_path:str,
    method:str,
    save_path:str = DATASET_PREPROCESSING_FOLDER,
    force_overwrite:bool = False):
    """
        Given a path to a dataset, it return, and save, the transformed dataset.

        :param dataset_path: Path where the dataset is. Datasets should be in a CSV format.
        :param method: Preprocessing method selected.
            It should be in
            ["min_max_scaler",
            "standard_scaler",
            "select_percentile",
            "pca",
            "fast_ica",
            "feature_agglomeration",
            "radial_basis_function_sampler"].
        :param save_path: The path were to save the new dataset.
        :param force_overwrite: If True force the overwrite of the file,
            otherwhise just don't do anything.

        :return: The transformed data.
    """

    if method not in LIST_OF_PREPROCESSING:
        raise CustomValueError(list_name='preprocessing', input_value=method)

    dataset_name = os.path.basename(dataset_path)

    logging.debug("The '%s' preprocessing has been selected for the '%s' dataset.",
    method, dataset_name)

    save_name = dataset_name #+ '.csv'
    directory = join(save_path, method)
    file_path = join(directory, save_name)

    # If file already created and force_overwrite is False
    # doesn't recompute the preprocessing but just read it from the file.
    if not force_overwrite and exists(file_path):
        new_data = read_csv(file_path)
        logging.debug("[skip] The '%s' preprocessing has already been created for the '%s' dataset."
            ,method, dataset_name)
        #remove last column
        transformed_data = new_data.iloc[:,:-1]
        return transformed_data.to_numpy()

    dataset = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset)

    if "y" in list(dataset.columns):
        y_data = dataset["y"].to_numpy()
        x_data  = dataset.drop(["y"], axis=1).to_numpy()
    else:
        y_data = dataset.iloc[: , -1].to_numpy()
        x_data = dataset.iloc[: , :-1].to_numpy()

    # Preprocessing
    transformed_data = preprocessing(method, x_data, y_data)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save of the transformed data
    new_data = np.concatenate((transformed_data, np.asarray([y_data]).T), axis=1)
    np.savetxt(file_path, new_data, delimiter=",")

    return transformed_data

def preprocessing(method:str, x_data, y_data):
    """
        Given a preprocessing method and data, it returns transformed data.

        :param method: Preprocessing method selected.
         It should be in
         ["min_max_scaler",
         "standard_scaler",
         "select_percentile",
         "pca",
         "fast_ica",
         "feature_agglomeration",
         "radial_basis_function_sampler"].
        :param x_data: Input variables

        :return: The transformed data.
    """
    if method not in LIST_OF_PREPROCESSING:
        raise CustomValueError(list_name='preprocessing', input_value=method)

    if method == 'min_max_scaler':
        transformed_data = min_max_scaler(x_data)
    elif method == 'standard_scaler':
        transformed_data = standard_scaler(x_data)
    elif method == 'select_percentile':
        transformed_data = select_percentile(x_data, y_data)
    elif method == 'pca':
        transformed_data = pca(x_data)
    elif method == 'fast_ica':
        n_components = min(100, int( x_data.shape[1] / 2 ))
        transformed_data = fast_ica(x_data, n_components)
    elif method == 'feature_agglomeration':
        n_clusters = min(100, int( x_data.shape[1] / 2 ))
        transformed_data = feature_agglomeration(x_data, n_clusters)
    elif method == 'radial_basis_function_sampler':
        transformed_data = radial_basis_function_sampler(x_data)

    return transformed_data

def categorical_string_to_number(dataset:pd.DataFrame) -> pd.DataFrame:
    """
        Given data it return the transformed data
         after replacing categorical string into numbers.

        :param dataset: A dataset, should be a dataframe.

        :return: The transformed data.
    """
    dataset_copy = dataset.copy(deep=True)
    for column in dataset_copy:
        # If the column type is not number
        if not is_numeric_dtype( dataset_copy[column] ):
            dataset_copy[column] = dataset_copy[column].astype('category').cat.codes
    return dataset_copy

def categorical_algorithm_to_number(dataset:pd.DataFrame) -> pd.DataFrame:
    """
        Given data it return the transformed data
         after replacing categorical string into numbers.

        :param dataset: A dataset, should be a dataframe.

        :return: The transformed data.
    """
    dataset_copy = dataset.copy(deep=True)
    dataset_copy.replace(
        LIST_OF_ML_MODELS,
        CATEGORICAL_LIST_OF_ML_MODELS,
        inplace=True)
    return dataset_copy

def min_max_scaler(x_data):
    """
        Given x_data it return the transformed data
         after the min max scaler.

        :param x_data: Input variables

        :return: The transformed data.
    """
    scaler = MinMaxScaler().fit(x_data)
    return scaler.transform(x_data)

def standard_scaler(x_data):
    """
        Given x_data it return the transformed data
         after the standard scaler.

        :param x_data: Input variables

        :return: The transformed data.
    """
    scaler = StandardScaler().fit(x_data)
    return scaler.transform(x_data)

def select_percentile(x_data, y_data, perc:int=10):
    """
        Given x_data it return the transformed data
         after selecting a percentile.

        :param x_data: Input variables

        :return: The transformed data.
    """
    # Shift all values to avoid negative number, since we are using chi2
    positive_x_data = x_data - np.amin(x_data)
    new_data = SelectPercentile(chi2, percentile=perc).fit_transform(positive_x_data, y_data)
    # Force n_features to be at least 2
    if new_data.shape[1] < 2:
        # It's not sure it solves the problem
        # but there is a good probability
        n_original_features = positive_x_data.shape[1]
        new_percentile = 1 / (n_original_features / 2)
        new_percentile = (new_percentile * 100) +1
        new_percentile = max(new_percentile, 25)

        new_data = SelectPercentile(
            chi2,
            percentile=new_percentile
            ).fit_transform(positive_x_data, y_data)
    return new_data

def pca(x_data, n_components:float=0.85):
    """
        Given x_data it return the transformed data
         after the PCA.

        :param x_data: Input variables

        :return: The transformed data.
    """
    transformer = PCA(
        n_components=n_components,
        svd_solver='full').fit(x_data)
    new_data = transformer.transform(x_data)

    # Force n_features to be at least 2
    if new_data.shape[1] < 2:
        transformer = PCA(
            n_components=2,
            svd_solver='full').fit(x_data)
        new_data = transformer.transform(x_data)

    return new_data

def fast_ica(x_data, n_components:None or int):
    """
        Given x_data it return the transformed data
         after the Fast ICA.

        :param x_data: Input variables

        :return: The transformed data.
    """
    transformer = FastICA(
        n_components=n_components,
        random_state=SEED_VALUE,
        whiten='unit-variance',
        max_iter= 10000,
        tol= 0.3 , # 0.001
        ).fit(x_data)
    return transformer.transform(x_data)

def feature_agglomeration(x_data, n_clusters:int):
    """
        Given x_data it return the transformed data
         after the Feature Agglomeration.

        :param x_data: Input variables

        :return: The transformed data.
    """

    agglo = cluster.FeatureAgglomeration(n_clusters=n_clusters).fit(x_data)
    return agglo.transform(x_data)

def radial_basis_function_sampler(x_data, gamma:int=1):
    """
        Given x_data it return the transformed data
         after the Radial Basis Function Sampler.

        :param x_data: Input variables

        :return: The transformed data.
    """
    rbf_feature = RBFSampler(gamma=gamma, random_state=SEED_VALUE).fit(x_data)
    return rbf_feature.transform(x_data)
