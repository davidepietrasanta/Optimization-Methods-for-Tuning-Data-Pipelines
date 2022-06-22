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
from os.path import isfile, join
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import cluster
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from ..config import SEED_VALUE, DATASET_PREPROCESSING_FOLDER, LIST_OF_PREPROCESSING # pylint: disable=relative-beyond-top-level

def preprocess_all_datasets(datasets_path, save_path=DATASET_PREPROCESSING_FOLDER, verbose=False):
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
         "polynomial_features",
         "radial_basis_function_sampler"].
        :param verbose: If True more info are printed.

        :return: List of sucessfully preprocessed datasets.
    """
    list_datasets = [f for f in listdir(datasets_path) if isfile(join(datasets_path, f))]
    n_datasets = len(list_datasets)
    preprocessed_datasets = []

    for i, dataset_name in enumerate(list_datasets):
        if verbose:
            print( "Transforming dataset: '" + dataset_name +
            "'...("+ str(i+1) + "/" + str(n_datasets) + ")")

        dataset_path = join(datasets_path, dataset_name)
        n_methods = 0
        for method in LIST_OF_PREPROCESSING:
            try:
                preprocess_dataset(dataset_path, method, save_path)
                n_methods = n_methods + 1
            except:  # pylint: disable=bare-except
                if verbose:
                    print( "Error, '" + method + "' skipped in '" + dataset_name + "'.")

        if n_methods == len(LIST_OF_PREPROCESSING):
            preprocessed_datasets.append(dataset_name)

    return preprocessed_datasets

def preprocess_dataset(dataset_path, method, save_path=DATASET_PREPROCESSING_FOLDER, verbose=False):
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
         "polynomial_features",
         "radial_basis_function_sampler"].
        :param save_path: The path were to save the new dataset.
        :param verbose: If True more info are printed.

        :return: The transformed data.
    """
    transformed_data = None

    if method in LIST_OF_PREPROCESSING:
        dataset_name = os.path.basename(dataset_path)

        if verbose:
            print("The '" + method +
            "' preprocessing has been selected for the '" + dataset_name + "' dataset.")

        dataset = pd.read_csv(dataset_path)
        dataset = categorical_string_to_number(dataset)

        y_data = dataset["y"].to_numpy()
        x_data  = dataset.drop(["y"], axis=1).to_numpy()

        # Preprocessing
        if verbose:
            print("Preprocessing...")
        transformed_data = preprocessing(method, x_data, y_data)

        # Save of the transformed data
        if verbose:
            print("Saving the transformed data...")

        save_name = dataset_name #+ '.csv'
        directory = join(save_path, method)
        file_path = join(directory, save_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        new_data = np.concatenate((transformed_data, np.asarray([y_data]).T), axis=1)
        np.savetxt(file_path, new_data, delimiter=",")

    else:
        if verbose:
            print("The preprocessing '" + method + "' is not between "
            +" ".join(LIST_OF_PREPROCESSING))

        return transformed_data

    return transformed_data

def preprocessing(method, x_data, y_data):
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
         "polynomial_features",
         "radial_basis_function_sampler"].
        :param x_data: Input variables

        :return: The transformed data.
    """

    transformed_data = None
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
    elif method == 'polynomial_features':
        transformed_data = polynomial_features(x_data)
    elif method == 'radial_basis_function_sampler':
        transformed_data = radial_basis_function_sampler(x_data)

    return transformed_data

def categorical_string_to_number(dataset):
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

def select_percentile(x_data, y_data, perc=10):
    """
        Given x_data it return the transformed data
         after selecting a percentile.

        :param x_data: Input variables

        :return: The transformed data.
    """
    # Shift all values to avoid negative number, since we are using chi2
    positive_x_data = x_data - np.amin(x_data)
    new_x_data = SelectPercentile(chi2, percentile=perc).fit_transform(positive_x_data, y_data)
    return new_x_data

def pca(x_data, n_components=0.85):
    """
        Given x_data it return the transformed data
         after the PCA.

        :param x_data: Input variables

        :return: The transformed data.
    """
    transformer = PCA(n_components=n_components, svd_solver='full').fit(x_data)
    return transformer.transform(x_data)

def fast_ica(x_data, n_components):
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

def feature_agglomeration(x_data, n_clusters):
    """
        Given x_data it return the transformed data
         after the Feature Agglomeration.

        :param x_data: Input variables

        :return: The transformed data.
    """

    agglo = cluster.FeatureAgglomeration(n_clusters=n_clusters).fit(x_data)
    return agglo.transform(x_data)

def polynomial_features(x_data, degree=2, interaction_only=True):
    """
        Given x_data it return the transformed data
         after the Polynomial Features transformation.

        :param x_data: Input variables

        :return: The transformed data.
    """
    poly = PolynomialFeatures(
        degree= degree,
        interaction_only=interaction_only
        ).fit(x_data)
    return poly.transform(x_data)

def radial_basis_function_sampler(x_data, gamma=1):
    """
        Given x_data it return the transformed data
         after the Radial Basis Function Sampler.

        :param x_data: Input variables

        :return: The transformed data.
    """
    rbf_feature = RBFSampler(gamma=gamma, random_state=SEED_VALUE).fit(x_data)
    return rbf_feature.transform(x_data)

# To DO:
# ADD IMAGE PREPROCESSING
# Should I detect if image or not first or maybe better not?
# Should make the previous function able to work also with images
#
# Data Aug
# Black and White
