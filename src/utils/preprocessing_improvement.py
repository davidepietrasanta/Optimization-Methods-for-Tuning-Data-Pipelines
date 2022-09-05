"""
    Module for the selection and download of the dataset to train the meta-learning model.
"""

from os import makedirs
from os.path import join, exists, dirname, basename
import logging
from operator import itemgetter
import numpy as np
from joblib import load


from src.config import LIST_OF_ML_MODELS, LIST_OF_PREPROCESSING, list_of_metafeatures
from src.config import METAFEATURES_MODEL_FOLDER, CATEGORICAL_LIST_OF_ML_MODELS
from src.exceptions import CustomValueError, exception_logging
from src.utils.preprocessing_methods import preprocess_dataset
from src.utils.metafeatures_extraction import metafeature
from src.utils.data_preparation import delta_funct


def predicted_improvement(
    dataset_path:str,
    algorithm:str,
    preprocessing:str = None,
    preprocessing_path:str = None,
    data_metafeatures = None,
    metalearner_path:str = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    ) -> float or None:
    """
        Predict the delta improvement of a dataset after the preprocessing.
        One between 'preprocessing' and 'preprocessing_path' must not be None.

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
        :param preprocessing: The preprocessing method to use.
        Must be in:
        [
            "min_max_scaler",
            "standard_scaler",
            "select_percentile",
            "pca",
            "fast_ica",
            "feature_agglomeration",
            "radial_basis_function_sampler"
        ]
        :param preprocessing_path: The path to the already preprocessing dataset
        , should be a csv file. \n
        If both 'preprocessing_path' and 'preprocessing' are not None 'preprocessing_path'
        has the priority.
        :param data_metafeatures: If you have already calculated the metafeatured
         of the raw dataset you can insert them here.
        :param metalearner_path: The path of the trained metalearner model, should be .joblib file.

        :return: The delta improvement.\n
         If None some error occurred during the metafeatures extraction or the preprocessing.
    """

    if preprocessing is None and preprocessing_path is None:
        raise ValueError("One between 'preprocessing' and 'preprocessing_path' must not be None.")

    if algorithm not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    logging.debug("Preprocessing the dataset: %s", dataset_path)
    preprocessing_path = _preprocess_dataset(dataset_path, preprocessing_path, preprocessing)
    logging.debug("Dataset preprocessed.")

    logging.debug("Extract metafeatures from raw dataset...")
    if data_metafeatures is None:
        data_metafeatures = metafeature(dataset_path)
        if data_metafeatures is None:
            logging.debug("Error while extracting the metafeatures from the raw dataset.")
            return None
    logging.debug("Extract metafeatures from preprocessed dataset...")
    preprocessed_data_metafeatures = metafeature(preprocessing_path)
    if preprocessed_data_metafeatures is None:
        logging.debug("Error while extracting the metafeatures from the preprocessed dataset.")
        return None
    logging.debug("Metafeatures Extracted from both datasets")


    prediction = _wrap_delta_and_prediction(
        data_metafeatures,
        preprocessed_data_metafeatures,
        algorithm,
        metalearner_path)

    if preprocessing is None:
        preprocessing = "Unknown"

    msg = str(
        "The delta improvement estimation for the dataset '"
        + basename(dataset_path)
        + "' with the preprocessing '" + preprocessing
        + "' and the algorithm '" + algorithm
        + "' is '" + str(prediction)
        +  "' [metalearner: '" + basename(metalearner_path) + "'])"
    )
    logging.info("%s", msg)
    return prediction

def ml_model_to_categorical(ml_model_name:str) -> int:
    """
        Convert the machine learning model name into a fixed integer.
        If the machine learning model is not known raise CustomValueError.
    """
    if ml_model_name not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=ml_model_name)

    index = LIST_OF_ML_MODELS.index(ml_model_name)
    return CATEGORICAL_LIST_OF_ML_MODELS[index]

def _preprocess_dataset(
    dataset_path,
    preprocessing_path,
    preprocessing
    ):
    """
        ciao
    """
    if preprocessing_path is not None:
        logging.debug("Dataset already preprocessed.")
    elif preprocessing is not None:
        logging.debug("Dataset NOT already preprocessed.")
        if preprocessing not in LIST_OF_PREPROCESSING:
            raise CustomValueError(list_name='preprocessing', input_value=preprocessing)


        preprocessing_path = dirname(dataset_path)
        if not exists(preprocessing_path):
            makedirs(preprocessing_path)

        try:
            _ = preprocess_dataset(
                dataset_path = dataset_path,
                method = preprocessing,
                save_path = preprocessing_path)
        except Exception: # pylint: disable=broad-except
            msg = "Error while preprocessing the dataset."
            exception_logging(msg)
            return None

        preprocessing_path = join(preprocessing_path, preprocessing)
        preprocessing_path = join(preprocessing_path, basename(dataset_path))
    return preprocessing_path

def _wrap_delta_and_prediction(
    data_metafeatures,
    preprocessed_data_metafeatures,
    algorithm,
    metalearner_path
    ):
    """
        Just a private function to wrap calculation of
         the delta and performances prediction.
         To avoid duplicated code.
    """
    logging.debug("Calculating the delta...")
    delta = _calculate_delta(data_metafeatures, preprocessed_data_metafeatures,algorithm)
    logging.debug("Delta calculated")

    logging.debug("Estimation of the delta...")
    logging.debug("Loading model: '%s'", metalearner_path)
    loaded_model = load(metalearner_path)
    logging.debug("Predicting...")
    prediction = loaded_model.predict(delta)
    prediction = prediction[0]
    return prediction

def _calculate_delta(
    data_metafeatures:str,
    preprocessed_data_metafeatures:str,
    algorithm:str
    ) -> np.array:
    """
        Private function to calculate delta.
    """
    metafeatures_list = list_of_metafeatures()

    non_preprocessed = []
    preprocessed = []

    for metafeatures in metafeatures_list:
        logging.debug("key: '%s', data='%f', preprocessed='%f'",
        metafeatures, data_metafeatures[metafeatures], preprocessed_data_metafeatures[metafeatures])

        non_preprocessed.append(data_metafeatures[metafeatures])
        preprocessed.append(preprocessed_data_metafeatures[metafeatures])

    delta = delta_funct(preprocessed, non_preprocessed)
    delta = np.insert(delta, 0, ml_model_to_categorical(algorithm))
    delta = delta.reshape(1, -1) # reshape 2D
    return delta

def max_in_dict(dict_data:dict) -> list:
    """
        Return the max value in the dict and the key associated to it.
        It doesn't consider None values.

        :param dict_data: A dict with None and numbers values.

        :return: The highest value and his key.
    """
    # Remove None values
    dict_data = {k: v for k, v in dict_data.items() if v is not None}
    # Search for the max value
    best = max(dict_data.items(), key=itemgetter(1))[0]
    return [best, dict_data[best]]

def best_one_step_bruteforce(
    dataset_path:str,
    algorithm:str,
    metalearner_path:str = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    ):
    """
        Return the best one-step preprocessing method based on the delta improvement.
        It is a brute force search so it runs all the possible preprocessing methods.

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
        :param metalearner_path: The path of the trained metalearner model, should be .joblib file.

        :return: Best one-step preprocessing method
    """
    results = one_step_bruteforce(dataset_path, algorithm, metalearner_path)
    return max_in_dict(results)

def one_step_bruteforce(
    dataset_path:str,
    algorithm:str,
    metalearner_path:str = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    ) -> dict or None:
    """
        Return the results of all the one-step preprocessing method.

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
        :param metalearner_path: The path of the trained metalearner model, should be .joblib file.

        :return: Dict of results for each one-step preprocessing method.
    """
    results = {}

    # Extract metafeatures from the raw dataset
    data_metafeatures = metafeature(dataset_path)
    if data_metafeatures is None:
        logging.debug("Error while extracting the metafeatures from the raw dataset.")
        return None

    for preprocessing in LIST_OF_PREPROCESSING:
        results[preprocessing] = predicted_improvement(
            dataset_path = dataset_path,
            algorithm = algorithm,
            preprocessing = preprocessing,
            data_metafeatures = data_metafeatures,
            metalearner_path = metalearner_path
            )

    return results

def pipeline_experiments(
    dataset_path:str,
    algorithm:str,
    list_of_experiments: list,
    metalearner_path:str = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    ) -> dict:
    """
        Performs a series of different preprocessing experiments, returning performances.

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
        :param metalearner_path: The path of the trained metalearner model, should be .joblib file.

        :return: A dict with all the performances of the experiments saved.
    """
    results = {}

    if algorithm not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    if len(list_of_experiments) == 0:
        return results

    data_metafeatures = metafeature(dataset_path)
    if data_metafeatures is None:
        logging.debug("Error while extracting the metafeatures from the raw dataset.")
        return None

    for experiment in list_of_experiments:
        results[str(experiment)] = preprocessing_experiment(
            dataset_path,
            algorithm,
            experiment,
            data_metafeatures,
            metalearner_path
        )

    return results

def preprocessing_experiment(
    dataset_path:str,
    algorithm:str,
    experiment: list,
    data_metafeatures,
    metalearner_path:str = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    ) -> float or None:
    """
        Run one preprocessing experiment.

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
        :param data_metafeatures:  If you have already calculated the metafeatured
         of the raw dataset you can insert them here.
        :param metalearner_path: The path of the trained metalearner model, should be .joblib file.

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

    logging.info("Calculating preprocessed metafeatures...")
    preprocessed_data_metafeatures = metafeature(preprocessed_dataset_path)
    if preprocessed_data_metafeatures is None:
        logging.debug("Error while extracting the metafeatures from the preprocessed dataset.")
        return None

    prediction = _wrap_delta_and_prediction(
        data_metafeatures,
        preprocessed_data_metafeatures,
        algorithm,
        metalearner_path)

    msg = str(
        "The delta improvement estimation for the dataset '"
        + basename(dataset_path)
        + "' with the experiment '" + str(experiment)
        + "' and the algorithm '" + algorithm
        + "' is '" + str(prediction)
        +  "' [metalearner: '" + basename(metalearner_path) + "'])"
    )
    logging.info("%s", msg)

    return prediction
