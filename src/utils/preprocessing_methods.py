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
from os.path import join
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import cluster
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from ..config import SEED_VALUE, DATASET_FOLDER, LIST_OF_PREPROCESSING # pylint: disable=relative-beyond-top-level

def preprocessing_dataset(dataset_path, method, save_path = DATASET_FOLDER, verbose = False):
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

        y_cord = dataset["y"].to_numpy()
        x_cord = dataset.drop(["y"], axis=1).to_numpy()

        # Preprocessing
        if verbose:
            print("Preprocessing...")
        transformed_data = preprocessing(method, x_cord)

        # Save of the transformed data
        if verbose:
            print("Saving the transformed data...")

        save_name = dataset_name + '-' + method + '.csv'
        file_path = join(save_path, save_name)

        # concatenate transformed_data e y
        #a = np.asarray([ [1,2], [4,5], [7,8], [7,1] ])
        #y = np.asarray([ 1, 2, 3, 4 ])
        #y = np.asarray([y])
        #data = np.concatenate((a, y.T), axis=1)
        new_data = np.concatenate((transformed_data, np.asarray([y_cord]).T), axis=1)
        np.savetxt(file_path, new_data, delimiter=",")

    else:
        if verbose:
            print("The preprocessing '" + method + "' is not between "
            +" ".join(LIST_OF_PREPROCESSING))

        return transformed_data

    return transformed_data

def preprocessing(method, x_data):
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
        transformed_data = select_percentile(x_data)
    elif method == 'pca':
        transformed_data = pca(x_data)
    elif method == 'fast_ica':
        transformed_data = fast_ica(x_data)
    elif method == 'feature_agglomeration':
        transformed_data = feature_agglomeration(x_data)
    elif method == 'polynomial_features':
        transformed_data = polynomial_features(x_data)
    elif method == 'radial_basis_function_sampler':
        transformed_data = radial_basis_function_sampler(x_data)

    return transformed_data


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

def select_percentile(x_data, perc=10):
    """
        Given x_data it return the transformed data
         after selecting a percentile.

        :param x_data: Input variables

        :return: The transformed data.
    """
    new_x_data = SelectPercentile(chi2, percentile=perc).fit_transform(x_data)
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

# To DO:
# See if there's a fast and simple way to determine n_components
def fast_ica(x_data, n_components=7):
    """
        Given x_data it return the transformed data
         after the Fast ICA.

        :param x_data: Input variables

        :return: The transformed data.
    """
    transformer = FastICA(
        n_components=n_components,
        random_state=SEED_VALUE,
        whiten='unit-variance'
        ).fit(x_data)
    return transformer.transform(x_data)

# To DO:
# See if there's a fast and simple way to determine n_clusters
def feature_agglomeration(x_data, n_clusters=32):
    """
        Given x_data it return the transformed data
         after the Feature Agglomeration.

        :param x_data: Input variables

        :return: The transformed data.
    """

    agglo = cluster.FeatureAgglomeration(n_clusters=n_clusters).fit(x_data)
    return agglo.transform(x_data)

def polynomial_features(x_data, min_degree=2, max_degree=5, interaction_only=True):
    """
        Given x_data it return the transformed data
         after the Polynomial Features transformation.

        :param x_data: Input variables

        :return: The transformed data.
    """
    poly = PolynomialFeatures(
        degree= (min_degree, max_degree),
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
