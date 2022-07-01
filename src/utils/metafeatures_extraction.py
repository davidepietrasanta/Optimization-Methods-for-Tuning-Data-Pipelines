"""
    Module for the extraction of the metafeatures.
"""
import math
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import skdim
from pymfe.mfe import MFE
from ..config import METAFEATURES_FOLDER # pylint: disable=relative-beyond-top-level
from .dataset_selection import check_missing_values # pylint: disable=relative-beyond-top-level


def metafeatures_extraction_data(datasets_path, save_path = METAFEATURES_FOLDER,
 name_saved_csv = None, verbose = False):
    """
        Given a path to datasets, it return, and save, the matefeatures for each dataset.

        :param datasets_path: Path where the datasets are. Datasets should be in a CSV format.
        :param save_path: Path where the metafeatures are saved.
        :param name_saved_csv: The name of the csv file with all the metafeatures.
         If None the name is 'metafeatures.csv'.
        :param verbose: If True more info are printed.

        :return: Metafeatures, as pandas Dataframe.
    """
    list_datasets = [f for f in listdir(datasets_path) if isfile(join(datasets_path, f))]
    n_datasets = len(list_datasets)
    datasets_metafeatures = []

    missing_values_datasets = check_missing_values(datasets_path)

    for i, dataset_name in enumerate(list_datasets):
        if verbose:
            print( "Extracting metafeatures from '" + dataset_name +
            "'...("+ str(i+1) + "/" + str(n_datasets) + ")")

        dataset_path = join(datasets_path, dataset_name)
        if dataset_path not in missing_values_datasets:
            metafeatures_data = metafeature(dataset_path)
            if metafeatures_data is not None:
                metafeatures_data['dataset_name'] = dataset_name
                datasets_metafeatures.append( metafeatures_data )
        else:
            if verbose:
                print( "'" + dataset_name + "' skipped because of missing values.")


    dataset = pd.DataFrame(datasets_metafeatures)

    if name_saved_csv is None:
        name_saved_csv = "metafeatures_data.csv"

    dataset.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset.to_csv(join(save_path, name_saved_csv), index=True)

    return dataset


def metafeature(dataset_path, verbose=False):
    """
        Given a path to a single dataset it return the metafeatures of the dataset.

        :param dataset_path: Path where the dataset is.
         It should be in a CSV format.
        :param verbose: If True more info are printed.

        :return: Metafeatures, as Dict, or None.
    """
    try:
        # Read the CSV
        dataset = pd.read_csv(dataset_path)

        # Separate X from y
        if 'y' in list(dataset.columns):
            y = dataset["y"].to_list() # pylint: disable=invalid-name
            X = dataset.drop(["y"], axis=1).to_numpy() # pylint: disable=invalid-name
        else:
            # This is because we may want to work with pre-processed data
            # Preprocess can change che nature of the features so it's not
            # possible to keep the original features name.
            y = dataset.iloc[: , -1].tolist() # pylint: disable=invalid-name
            X = dataset.iloc[: , :-1].to_numpy() # pylint: disable=invalid-name

        # Extract general, statistical and information-theoretic measures
        mfe = MFE(groups=["general", "statistical", "info-theory"], suppress_warnings= not verbose)
        mfe.fit(X, y)
        features = mfe.extract(suppress_warnings= not verbose )
        keys = features[0]
        values = features[1]

        dict_ft = {}
        for i, key_i in enumerate(keys):
            # Not insert NaN values
            if not math.isnan(values[i]):
                dict_ft[key_i] = values[i]

        intrinsic_dim = intrinsic_dimensionality(X)
        dict_ft['intrinsic_dim.global'] = intrinsic_dim[0]
        dict_ft['intrinsic_dim.local.mean'] = intrinsic_dim[1]

        if verbose:
            print( str(len(dict_ft))
            + " meta-features were extracted from the dataset: "
            + str(dataset_path))

        return dict_ft

    except Exception: # pylint: disable=broad-except
        if verbose:
            print( "Error while extracting metafeature of '" + str(dataset_path) + "', skipped.")
        return None


def intrinsic_dimensionality(data):
    """
        Calculate the gloabal intrinsic dimensionality and the mean of
         the local intrinsic dimensionality.\n
        To know more visit: https://scikit-dimension.readthedocs.io/en/latest/api.html

        :param data: The data in a numpy format.

        :return: A list with the gloabal intrinsic dimensionality
         and the mean of the local intrinsic dimensionality.

    """
    #estimate global intrinsic dimension
    corr_int = skdim.id.CorrInt().fit(data)
    #estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
    lpca = skdim.id.lPCA().fit_pw(data,
                                n_neighbors = 100,
                                n_jobs = 1)

    #get estimated intrinsic dimension
    global_intrinsic_dimensionality = corr_int.dimension_
    mean_local_intrinsic_dimensionality = np.mean(lpca.dimension_pw_)

    intrinsic_dim = [global_intrinsic_dimensionality, mean_local_intrinsic_dimensionality]
    return intrinsic_dim

# TO DO:
# See how to integrate Akaike and Kl-divergence
#from sklearn.preprocessing import normalize
#def kl_divergence(distr_p, distr_q):
#    """
#        Simple implementation of the Kullback-Leibler divergence.
#
#        :param p: Distribution p.
#        :param q: Distribution q.
#
#        :example of use:
#            import numpy as np \n
#            from scipy.stats import norm \n
#            from matplotlib import pyplot as plt \n
#            import tensorflow as tf \n
#            import seaborn as sns \n
#            sns.set() \n
#            \n
#            x = np.arange(-10, 10, 0.001) \n
#            p = norm.pdf(x, 0, 2) \n
#            q = norm.pdf(x, 2, 2)plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q)) \n
#            plt.plot(x, p) \n
#            plt.plot(x, q, c='red') \n
#
#        :return: A scalar, the Killback-Leibler divergence between p and q.
#    """
#    # We first need to normalize the vectors to Probability density function
#    # The sum should be 1
#    p_norm = normalize(
#        distr_p[:,np.newaxis],
#        axis=0,
#        norm='l1').ravel()
#    q_norm = normalize(
#        distr_q[:,np.newaxis],
#        axis=0,
#        norm='l1').ravel()
#
#   return np.sum(np.where(p_norm != 0, p_norm * np.log(p_norm / q_norm), 0))

# TO DO:
# Should we use the Train or the whole dataset?
#def akaike(model, train_x, train_y):
#    """
#        Compute the The Akaike Information Criterion (AIC).\n
#        AIC = 2K - 2ln(L) \n
#        K is the Number of independent variables + 2 \n
#        L is the likelihood \n
#
#       :param model: A sklearn trained model
#        :param train_x: Training input variables
#       :param train_y: Training label or target value
#
#        :return: The Akaike Information Criterion
#    """
#    ## should be for the specific model we are doing!
#    return 0
 