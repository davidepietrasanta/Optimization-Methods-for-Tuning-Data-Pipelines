import math
import pandas as pd
import numpy as np
import skdim
from ..config import DATASET_FOLDER
from pymfe.mfe import MFE

# TO DO:
# ALL 
def metafeatures_extraction(datasets_path, save_path = DATASET_FOLDER, verbose=False):
    """
        Given a path to datasets, it return, and save, the matefeatures for each dataset.

        :param datasets_path: Path where the datasets are. 
        Datasets should be in a CSV format.
        :param save_path: Path where the metafeatures are saved.
        :param verbose: If True more info are printed.
        
        :return: Metafeatures, as pandas Dataframe.
    """ 
    
    return None


def metafeature(dataset_path, verbose=False):
    """
        Given a path to a single dataset it return the metafeatures of the dataset.

        :param dataset_path: Path where the dataset is. 
        It should be in a CSV format.
        :param verbose: If True more info are printed.
        
        :return: Metafeatures, as Dict.
    """ 
    # Read the CSV
    df = pd.read_csv(dataset_path)
    # Separate X from y
    y = df["y"].to_list()    
    X = df.drop(["y"], axis=1).to_numpy()

    # Extract general, statistical and information-theoretic measures
    mfe = MFE(groups=["general", "statistical", "info-theory"], suppress_warnings= not verbose)
    mfe.fit(X, y)
    ft = mfe.extract(suppress_warnings= not verbose )
    keys = ft[0]
    values = ft[1]

    dict_ft = {}
    for i in range(len(keys)):
        # Not insert NaN values
        if( not math.isnan(values[i]) ):
            dict_ft[keys[i]] = values[i]
    
    intrinsic_dim = intrinsic_dimensionality(X)
    dict_ft['intrinsic_dim.global'] = intrinsic_dim[0]
    dict_ft['intrinsic_dim.local.mean'] = intrinsic_dim[1]

    if verbose:
        print( str(len(dict_ft)) + " meta-features were extracted from the dataset: " + str(dataset_path))

    return dict_ft


def intrinsic_dimensionality(data):
    """
        Calculate the gloabal intrinsic dimensionality and the mean of the local intrinsic dimensionality.\n
        To know more visit: https://scikit-dimension.readthedocs.io/en/latest/api.html

        :param data: The data in a numpy format.

        :return: A list with the gloabal intrinsic dimensionality and the mean of the local intrinsic dimensionality.

    """
    #estimate global intrinsic dimension
    corrInt = skdim.id.CorrInt().fit(data)
    #estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
    lpca = skdim.id.lPCA().fit_pw(data,
                                n_neighbors = 100,
                                n_jobs = 1)

    #get estimated intrinsic dimension
    global_intrinsic_dimensionality = corrInt.dimension_
    mean_local_intrinsic_dimensionality = np.mean(lpca.dimension_pw_)

    intrinsic_dim = [global_intrinsic_dimensionality, mean_local_intrinsic_dimensionality]
    return intrinsic_dim

# TO DO:
# Both Kullback-Leibler divergence and AKAIKE Information Criterion has to be calculated after the model has fitted. 
def kl_divergence(p, q):
    """
        Simple implementation of the Kullback-Leibler divergence.

        :param p: Distribution p.
        :param q: Distribution q.

        :example of use:
            import numpy as np \n
            from scipy.stats import norm \n
            from matplotlib import pyplot as plt \n
            import tensorflow as tf \n
            import seaborn as sns \n
            sns.set() \n
            \n
            x = np.arange(-10, 10, 0.001) \n
            p = norm.pdf(x, 0, 2) \n
            q = norm.pdf(x, 2, 2)plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q)) \n
            plt.plot(x, p) \n
            plt.plot(x, q, c='red') \n
        
        :return: A scalar, the Killback-Leibler divergence between p and q.
    """ 
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# TO DO:
# Only if we use sklearn models
def AIC(model, data):
    """
        AIC = 2K - 2ln(L) \n
        K is the Number of independent variables + 2 \n
        L is the likelihood \n
    """
    return model.aic(data)
