"""
    Module for the selection and download of the dataset to train the meta-learning model.
"""

from os.path import join, dirname, basename
import logging
import time
import numpy as np


from src.config import LIST_OF_ML_MODELS, METAFEATURES_MODEL_FOLDER
from src.exceptions import CustomValueError
from src.utils.preprocessing_methods import preprocess_dataset
from src.utils.machine_learning_algorithms import machine_learning_algorithm
from src.utils.data_preparation import delta_funct
from src.utils.preprocessing_improvement import pipeline_experiments, max_in_dict


def pipeline_true_experiments(
    dataset_path:str,
    algorithm:str,
    list_of_experiments: list) -> dict:
    """
        Performs a series of different preprocessing experiments,
         returning the true performances.

        :param dataset_path: The path to the dataset, should be a csv file.
        :param algorithm: The machine learning algorithm you want to consider,
         should be in:
        [
            "logistic_regression",
            "naive_bayes",
            "knn",
            "random_forest",
            "svm",
            "perceptron"
        ]
        :param list_of_experiments: Should be a list of list preprocessing methods.

        :return: A dict with all the performances of the experiments saved.
    """
    results = {}

    if algorithm not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    if len(list_of_experiments) == 0:
        return results

    for experiment in list_of_experiments:
        results[str(experiment)] = preprocessing_pipeline(
            dataset_path,
            algorithm,
            experiment)

    logging.info("Calculating performances without preprocessing...")
    [_, prediction] = machine_learning_algorithm(
        dataset_path = dataset_path,
        algorithm = algorithm,
        cross_validation = True)

    results['None'] = prediction['f1_score']

    msg = str(
        "The performances for the dataset '"
        + basename(dataset_path)
        + "' without preprocessing"
        + "' and with the algorithm '" + algorithm
        + "' are '" + str(results['None'])
    )
    logging.info("%s", msg)

    logging.info("Calculating the true delta...")
    delta_results = _true_delta(results)
    logging.info("True delta calculated")

    return [results, delta_results]

def preprocessing_pipeline(
    dataset_path:str,
    algorithm:str,
    experiment: list) -> float or None:
    """
        Run one preprocessing experiment and return the performance of the
         algorithm selected.

        :param dataset_path: The path to the dataset, should be a csv file.
        :param algorithm: The machine learning algorithm you want to consider,
         should be in:
        [
            "logistic_regression",
            "naive_bayes",
            "knn",
            "random_forest",
            "svm",
            "perceptron"
        ]
        :param experiment: Should be a list preprocessing methods.

        :return: Estimated performances after the preprocessing experiment.
    """
    # Run all the preprocessing experiments
    logging.info("Preprocessing experiment...")
    preprocessed_dataset_path = dataset_path
    for preprocessing in experiment:
        logging.info("Calculating preprocessed metafeatures...")
        preprocess_dataset(
            dataset_path = preprocessed_dataset_path,
            method = preprocessing,
            save_path = dirname(preprocessed_dataset_path)
            )

        save_preprocessed = join(dirname(preprocessed_dataset_path),preprocessing)
        preprocessed_dataset_path = join(save_preprocessed,basename(dataset_path))

    logging.info("Calculating preprocessed performances...")

    [_, prediction] = machine_learning_algorithm(
        dataset_path = preprocessed_dataset_path,
        algorithm = algorithm,
        cross_validation = True
    )

    # f1_score
    performances = prediction['f1_score']

    msg = str(
        "The performances for the dataset '"
        + basename(dataset_path)
        + "' with the experiment '" + str(experiment)
        + "' and the algorithm '" + algorithm
        + "' are '" + str(performances)
    )
    logging.info("%s", msg)

    return performances

def _true_delta(results:dict) -> dict:
    """

    """
    delta_dict = {}
    for key in results:
        delta = delta_funct(
            preprocessed = np.array([results[key]]),
            non_preprocessed = np.array([results['None']]) )

        delta_dict[key] = delta[0]

    return delta_dict


def experiment_on_dataset(
    dataset_path:str,
    algorithm:str,
    list_of_experiments:list) -> dict:
    """
        He performs an experiment where he searches, with each metalearner,
         for the best preprocessing and then compares it to the true optimal.

        :param dataset_path: The path to the dataset, should be a csv file.
        :param algorithm: The machine learning algorithm you want to consider,
         should be in:
        [
            "logistic_regression",
            "naive_bayes",
            "knn",
            "random_forest",
            "svm",
            "perceptron"
        ]
        :param list_of_experiments: Should be a list of list preprocessing methods.

        :return: A dict with all the performances/delta.
    """
    results = {}
    results['time(s)'] = {}

    start = time.time()
    results['gp'] = pipeline_experiments(
        dataset_path = dataset_path,
        algorithm = algorithm,
        list_of_experiments = list_of_experiments,
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_gaussian_process.joblib')
        )
    results['time(s)']['gp'] = time.time() - start

    logging.info("Estimations with Gaussian Process model")
    logging.info("%s", str(results['gp']))

    start = time.time()
    results['rf'] = pipeline_experiments(
        dataset_path = dataset_path,
        algorithm = algorithm,
        list_of_experiments = list_of_experiments,
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
        )
    results['time(s)']['rf'] = time.time() - start

    logging.info("Estimations with Random Forest model")
    logging.info("%s", str(results['rf']))

    start = time.time()
    results['knn'] = pipeline_experiments(
        dataset_path = dataset_path,
        algorithm = algorithm,
        list_of_experiments = list_of_experiments,
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_knn.joblib')
        )
    results['time(s)']['knn'] = time.time() - start

    logging.info("Estimations with KNN model")
    logging.info("%s", str(results['knn']))

    [key_gp, value_gp] = max_in_dict(results['gp'])
    logging.info("The best experiment for GP is %s, with %s estimated improvement.",
     str(key_gp), str(value_gp))

    [key_rf, value_rf] = max_in_dict(results['rf'])
    logging.info("The best experiment for RF is %s, with %s estimated improvement.",
     str(key_rf), str(value_rf))

    [key_knn, value_knn] = max_in_dict(results['knn'])
    logging.info("The best experiment for KNN is %s, with %s estimated improvement.",
     str(key_knn), str(value_knn))

    start = time.time()
    [results['true_results'], results['true_delta']] = pipeline_true_experiments(
        dataset_path = dataset_path,
        algorithm = algorithm,
        list_of_experiments = list_of_experiments)
    results['time(s)']['true'] = time.time() - start

    logging.info("True performances")
    logging.info("%s", str(results['true_results']) )
    logging.info("True Delta")
    logging.info("%s", str(results['true_delta']) )

    [key_true, value_true] = max_in_dict(results['true_results'])

    summary = str(
        "The best experiment is " + str(key_true)
        + " with " + str(value_true) + " of f1-score. "
        + "We have an estimated solution of " + str(results['true_results'][key_gp])
        +" (" + str(key_gp) + ") with GP,"
        +"a solution of " + str(results['true_results'][key_rf])
        +" (" + str(key_rf) + ") with RF and "
        +"a solution of "+ str(results['true_results'][key_knn])
        +" (" + str(key_knn) + ") with KNN. "
        +"While the optimal solution is " + str(value_true)
        + " (" + str(key_true)  + ")."
        )

    logging.info(summary)

    results['best'] = {
        'gp': [key_gp, results['true_results'][key_gp]],
        'rf': [key_rf, results['true_results'][key_rf]],
        'knn': [key_knn, results['true_results'][key_knn]],
        'real': [key_true, value_true]
    }

    results['summary'] = summary

    return results
