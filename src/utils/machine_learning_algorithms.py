"""
    Module for the machine learning algorithm.
    It implement the following algorithms:\n
    - Logistic Regression\n
    - Naive Bayes\n
    - K-NN\n
    - Random Forest\n
    - SVM\n
    - Perceptron\n
"""
import os
from os import listdir
from os.path import isfile, join, exists
import logging
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score

from src.config import LIST_OF_ML_MODELS, MODEL_FOLDER
from src.config import METAFEATURES_FOLDER
from src.config import SEED_VALUE, TEST_SIZE
from src.exceptions import CustomValueError, exception_logging
from .preprocessing_methods import categorical_string_to_number

def extract_machine_learning_performances(
    datasets_path:str,
    save_model_path:str = MODEL_FOLDER,
    save_performance_path:str = METAFEATURES_FOLDER,
    performance_file_name:str = 'performances.csv',
    preprocessing:None or str = None):
    """
        Given a path to a dataset and an algorithm, it return, and save,
         the trained model and the performance.

        :param datasets_path: Path where the dataset is. Datasets should be in a CSV format.
        :param save_model_path: The path were to save the trained model.
        :param save_performance_path: The path were to save the performances.
        :param performance_file_name: The name of the performances file (should be .csv)
        :param preprocessing: Name of the preprocessing choosen.

        :return: The performances.
    """
    list_datasets = sorted([f for f in listdir(datasets_path) if isfile(join(datasets_path, f))])

    performances = {}
    if preprocessing is not None:
        performances['preprocessing']  = preprocessing
    else:
        performances['preprocessing']  = 'None'

    if not os.path.exists(save_performance_path):
        os.makedirs(save_performance_path)

    perf_path = join(save_performance_path, performance_file_name)

    for j, dataset_name in enumerate(list_datasets):
        logging.debug("Dataset: '%s'...(%d/%d)",
            dataset_name, j+1, len(list_datasets) )

        for i, algorithm in enumerate(LIST_OF_ML_MODELS):
            logging.info(
                "Extracting performance from: '%s' [%d/%d] with '%s' and '%s'...(%d/%d)",
                dataset_name, j+1, len(list_datasets),
                algorithm, preprocessing,
                i+1,len(LIST_OF_ML_MODELS)
                )

            # If the metafeatures are already been extracted we don't do it again
            if not _check_csv_ml_algorithm(perf_path, dataset_name, preprocessing, algorithm):
                # Extract performances
                try:
                    [_, performance] = machine_learning_algorithm(
                        join(datasets_path, dataset_name),
                        algorithm,
                        save_path = save_model_path)

                    performances['dataset_name'] = dataset_name
                    performances['algorithm'] = algorithm
                    performances['performance'] = performance

                    performances_df = pd.DataFrame.from_dict([performances])
                    # transorm dict to str
                    performances_df['performance'] = performances_df['performance'].astype(str)

                    # If the file already exist it append the information
                    if exists(perf_path):
                        dataset_df = pd.read_csv(perf_path)
                        performances_df = dataset_df.merge(performances_df, how='outer')

                    performances_df.to_csv(
                        perf_path,
                        index=False)

                    logging.info("'%s' metafeatures saved in csv.", dataset_name)

                except Exception: # pylint: disable=broad-except
                    exception_logging(
                        "Error while extracting performance from '"+
                        dataset_name + "' with '" + algorithm + "' and '" +
                        preprocessing + "', skipped."
                    )

            else:
                logging.info("'%s' with '%s' and '%s' skipped because of already in csv.",
                             dataset_name, preprocessing, algorithm)

    if exists(perf_path):
        dataset_df = pd.read_csv(perf_path)
    else:
        dataset_df = None

    return dataset_df

def machine_learning_algorithm(
    dataset_path:str,
    algorithm:str,
    save_path:str = MODEL_FOLDER,
    cross_validation:bool = False):
    """
        Given a path to a dataset and an algorithm, it return, and save,
         the trained model and the performance.

        :param dataset_path: Path where the dataset is. Datasets should be in a CSV format.
        :param algorithm: Machine Learning algorithm selected.
         It should be in
         ["logistic_regression",
         "naive_bayes",
         "knn",
         "random_forest",
         "svm",
         "perceptron"].
        :param save_path: The path were to save the trained model.
        :param cross_validation: Train using a 10-fold cross validation.

        :return: A trained model with the performance.
    """
    if algorithm not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    dataset_name = os.path.basename(dataset_path)

    logging.debug("The '%s' algorithm has been selected for the '%s' dataset.",
        algorithm, dataset_name )

    dataset = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset)

    train, test = train_test_split(dataset, test_size=TEST_SIZE, random_state=SEED_VALUE)

    if "y" in list(dataset.columns):
        train_y = train["y"].to_numpy()
        train_x = train.drop(["y"], axis=1).to_numpy()

        test_y = test["y"].to_numpy()
        test_x = test.drop(["y"], axis=1).to_numpy()
    else:
        train_y = train.iloc[: , -1].tolist()
        train_x = train.iloc[: , :-1].to_numpy()

        test_y = test.iloc[: , -1].tolist()
        test_x = test.iloc[: , :-1].to_numpy()

    # Train
    logging.debug("Training '%s'...", algorithm)
    model = train_algorithm(algorithm, train_x, train_y)

    # Save of the model
    logging.debug("Saving the trained model...")

    directory = join(save_path, algorithm)
    file_path = join(
        directory,
        dataset_name + '-' + algorithm + '.joblib')

    if not os.path.exists(directory):
        os.makedirs(directory)

    dump(model, file_path)

    # Test
    logging.debug("Testing...")
    if cross_validation:
        prediction = cross_validation_score(model, dataset)
        logging.debug("Prediction performance with 10-fold cv: %s", prediction)
    else:
        prediction = prediction_metrics(model, test_x, test_y)
        logging.debug("Prediction performance on test set: %s", prediction)

    return [model, prediction]

def train_algorithm(algorithm:str, train_x, train_y):
    """
        Given a ML algorithm and training data, it returns a trained model.

        :param algorithm: Machine Learning algorithm selected.
         It should be in
         ["logistic_regression",
         "naive_bayes",
         "knn",
         "random_forest",
         "svm",
         "perceptron"].
        :param train_x: Training input variables
        :param train_y: Training label or target value

        :return: A trained model.
    """

    if algorithm == 'logistic_regression':
        model = logistic_regression(train_x, train_y)
    elif algorithm == 'naive_bayes':
        model = naive_bayes(train_x, train_y)
    elif algorithm == 'knn':
        n_classes = len( set(train_y) )
        model = knn(train_x, train_y, n_classes)
    elif algorithm == 'random_forest':
        model = random_forest(train_x, train_y)
    elif algorithm == 'svm':
        model = svm(train_x, train_y)
    elif algorithm == 'perceptron':
        model = perceptron(train_x, train_y)
    else:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    return model

def cross_validation_score(model, dataset):
    """
        Return f1_score of the model calculated with 10-fold cross validation.
        Only for classification model.

        :param model: A trained sklearn model
        :param dataset: Test input variables

        :return: The mean of the f1_score of the model calculated with 10-fold cross validation.
    """

    if "y" in list(dataset.columns):
        dataset_y = dataset["y"].to_numpy()
        dataset_x = dataset.drop(["y"], axis=1).to_numpy()
    else:
        dataset_y = dataset.iloc[: , -1].tolist()
        dataset_x = dataset.iloc[: , :-1].to_numpy()

    scores = cross_val_score(model, dataset_x, dataset_y, cv=10, scoring='f1_macro')
    prediction = {
        "f1_score_all" : scores,
        "f1_score" : scores.mean(),
        "cv_std" : scores.std(),
        "cv_mean" : scores.mean(),
    }

    return prediction

def logistic_regression(x_train, y_train):
    """
        Given X and y return a trained Logistic Regression model.

        :param x_train: Input variables
        :param y_train: Label or Target value

        :return: A trained model.
    """
    model = LogisticRegression(random_state=SEED_VALUE, max_iter=10000).fit(x_train, y_train)
    return model

def naive_bayes(x_train, y_train):
    """
        Given X and y return a trained Naive Bayes model.

        :param x_train: Input variables
        :param y_train: Label or Target value

        :return: A trained model.
    """
    model = GaussianNB().fit(x_train, y_train)
    return model

def knn(x_train, y_train, n_neighbors:int):
    """
        Given X and y return a trained K-Neighbors model.

        :param x_train: Input variables
        :param y_train: Label or Target value
        :param n_neighbors: Number of neighbors to consider

        :return: A trained model.
    """
    # This is to avoid the following error:
    # "BLAS : Program is Terminated.
    # Because you tried to allocate too many memory regions."
    # That arise when running KNN on a pc with high number of jobs.
    n_jobs = 4

    model = KNeighborsClassifier(n_neighbors, n_jobs=n_jobs).fit(x_train, y_train)
    return model

def random_forest(x_train, y_train):
    """
        Given X and y return a trained Random Forest model.

        :param x_train: Input variables
        :param y_train: Label or Target value

        :return: A trained model.
    """
    model = RandomForestClassifier(random_state=SEED_VALUE).fit(x_train, y_train)
    return model

def svm(x_train, y_train):
    """
        Given X and y return a trained Support Vector Machine model.

        :param x_train: Input variables
        :param y_train: Label or Target value

        :return: A trained model.
    """
    model = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(x_train, y_train)
    return model

def perceptron(x_train, y_train):
    """
        Given X and y return a trained Perceptron model.

        :param x_train: Input variables
        :param y_train: Label or Target value

        :return: A trained model.
    """
    model = Perceptron(tol=1e-3, random_state=SEED_VALUE).fit(x_train, y_train)
    return model

def prediction_metrics(
    model,
    test_x,
    test_y,
    metrics:None or list = None,
    regression:bool = False) -> dict:
    """
        Return accuracy, precision, recall and f1_score of the model.

        :param model: A trained sklearn model
        :param test_x: Test input variables
        :param test_x: Test label or Target value
        :param metrics: List of metrics you want to calculate.
         If [] empty or None it calculates all.
         The metrics that can be calculated are:
         ["accuracy",
         "precision",
         "recall",
         "f1_score"].
        :param regression: If True return metrics for regression

        :return: Accuracy, precision, recall and f1_score as a dictionary.
    """
    model.n_jobs = 2
    prediction_test_y = model.predict(test_x)

    metrics_values = {}
    if not regression:
        if (metrics is None) or ('accuracy' in metrics):
            metrics_values["accuracy"] = accuracy_score(test_y, prediction_test_y)

        if (metrics is None) or ('precision' in metrics):
            metrics_values["precision"]=precision_score(test_y, prediction_test_y, average='micro')

        if (metrics is None) or ('recall' in metrics):
            metrics_values["recall"] = recall_score(test_y, prediction_test_y, average='micro')

        if (metrics is None) or ('f1_score' in metrics):
            metrics_values["f1_score"] = f1_score(test_y, prediction_test_y, average='micro')
    else:
        if (metrics is None) or ('mse' in metrics):
            metrics_values["mse"] = mean_squared_error(test_y, prediction_test_y)
        if (metrics is None) or ('rmse' in metrics):
            metrics_values["rmse"] = mean_squared_error(test_y, prediction_test_y, squared=False)
        if (metrics is None) or ('mae' in metrics):
            metrics_values["mae"] = mean_absolute_error(test_y, prediction_test_y)

    return metrics_values

def _check_csv_ml_algorithm(
    csv_path:str,
    dataset_name:str,
    preprocessing:str,
    algorithm:str
    ) -> bool:
    """
        Check if in the csv file there is line with same
        'dataset_name', 'preprocessing' and 'algorithm'.

        :param csv_path: Path of the csv file.
        :param dataset_name: Name of the dataset you are looking for.
        :param preprocessing: Name of the preprocessing you are looking for.
        :param algorithm: Name of the algorithm you are looking for.

        :return: True if 'dataset_name', 'preprocessing' and 'algorithm' already in the csv.
    """
    # Open the csv
    if not exists(csv_path):
        return False
    csv_data = pd.read_csv(csv_path)

    rows = csv_data.loc[
        (csv_data['dataset_name'] == dataset_name) &
        (csv_data['preprocessing'] == preprocessing) &
        (csv_data['algorithm'] == algorithm)]

    return len(rows.index) > 0
