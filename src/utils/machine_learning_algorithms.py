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
from os.path import isfile, join
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

from ..config import LIST_OF_ML_MODELS, MODEL_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import METAFEATURES_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import SEED_VALUE, TEST_SIZE # pylint: disable=relative-beyond-top-level
from ..exceptions import CustomValueError # pylint: disable=relative-beyond-top-level

def extract_machine_learning_performances(
    datasets_path,
    save_model_path = MODEL_FOLDER,
    save_performance_path = METAFEATURES_FOLDER,
    performance_file_name = 'performances.csv',
    preprocessing = None,
    verbose = False):
    """
        Given a path to a dataset and an algorithm, it return, and save,
         the trained model and the performance.

        :param dataset_path: Path where the dataset is. Datasets should be in a CSV format.
        :param save_path: The path were to save the trained model.
        :param verbose: If True more info are printed.

        :return: A trained model with the performance.
    """
    list_datasets = [f for f in listdir(datasets_path) if isfile(join(datasets_path, f))]

    performances = {
        'dataset_name' : [],
        'algorithm': [],
        'performance' : []
    }

    if preprocessing is not None:
        performances['preprocessing'] = []

    for i, dataset_name in enumerate(list_datasets):
        if verbose:
            print( "Dataset: '" + dataset_name +
            "'...("+ str(i+1) + "/" + str( len(list_datasets) ) + ")")

        dataset_path = join(datasets_path, dataset_name)

        for algorithm in LIST_OF_ML_MODELS:

            if verbose:
                print("Extracting performance from '" + dataset_name +
                "' with '" + algorithm + "'.")
            try:
                [_, performance] = machine_learning_algorithm(
                    dataset_path,
                    algorithm,
                    save_path = save_model_path,
                    verbose = False)

                performances['dataset_name'].append(dataset_name)
                performances['algorithm'].append(algorithm)
                performances['performance'].append(performance)
                if preprocessing is not None:
                    performances['preprocessing'].append(preprocessing)
            except Exception: # pylint: disable=broad-except
                if verbose:
                    print( "Error while extracting performance from '"
                    + dataset_name + "' with '" + algorithm + "', skipped.")

    save_path = save_performance_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    perf_path = join(save_path, performance_file_name)
    pd.DataFrame.from_dict(performances).to_csv(perf_path)

    return performances

def machine_learning_algorithm(dataset_path, algorithm, save_path = MODEL_FOLDER, verbose = False):
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
        :param verbose: If True more info are printed.

        :return: A trained model with the performance.
    """
    if algorithm not in LIST_OF_ML_MODELS:
        raise CustomValueError(list_name='ml_models', input_value=algorithm)

    dataset_name = os.path.basename(dataset_path)

    if verbose:
        print("The '" + algorithm +
        "' algorithm has been selected for the '" + dataset_name + "' dataset.")

    dataset = pd.read_csv(dataset_path)

    train, test = train_test_split(dataset, test_size=TEST_SIZE, random_state=SEED_VALUE)

    if 'y' in list(dataset.columns):
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
    if verbose:
        print("Training...")
    model = train_algorithm(algorithm, train_x, train_y)

    # Save of the model
    if verbose:
        print("Saving the trained model...")

    save_name = dataset_name + '-' + algorithm + '.joblib'
    directory = join(save_path, algorithm)
    file_path = join(directory, save_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    dump(model, file_path)

    # Test
    if verbose:
        print("Testing...")

    prediction = prediction_metrics(model, test_x, test_y)
    if verbose:
        print("Prediction performance on test set:")
        print(prediction)

    return [model, prediction]

def train_algorithm(algorithm, train_x, train_y):
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

    model = None
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


def logistic_regression(X, y): # pylint: disable=invalid-name
    """
        Given X and y return a trained Logistic Regression model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = LogisticRegression(random_state=SEED_VALUE, max_iter=10000).fit(X, y)
    return model

def naive_bayes(X, y): # pylint: disable=invalid-name
    """
        Given X and y return a trained Naive Bayes model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = GaussianNB().fit(X, y)
    return model

def knn(X, y, n_neighbors): # pylint: disable=invalid-name
    """
        Given X and y return a trained K-Neighbors model.

        :param X: Input variables
        :param y: Label or Target value
        :param n_neighbors: Number of neighbors to consider

        :return: A trained model.
    """
    model = KNeighborsClassifier(n_neighbors).fit(X, y)
    return model

def random_forest(X, y): # pylint: disable=invalid-name
    """
        Given X and y return a trained Random Forest model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = RandomForestClassifier(random_state=SEED_VALUE).fit(X, y)
    return model

def svm(X, y): # pylint: disable=invalid-name
    """
        Given X and y return a trained Support Vector Machine model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X, y)
    return model

def perceptron(X, y): # pylint: disable=invalid-name
    """
        Given X and y return a trained Perceptron model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = Perceptron(tol=1e-3, random_state=SEED_VALUE).fit(X, y)
    return model

def prediction_metrics(model, test_x, test_y, metrics = None, regression = False):
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
