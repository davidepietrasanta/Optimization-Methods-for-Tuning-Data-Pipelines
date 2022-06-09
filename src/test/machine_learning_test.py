"""
    Module for testing preprocessing methods.
"""
import shutil
from os.path import join, exists
from sklearn.model_selection import train_test_split
import pandas as pd
from ..utils.machine_learning_algorithms import machine_learning_algorithm # pylint: disable=relative-beyond-top-level
from ..utils.machine_learning_algorithms import train_algorithm # pylint: disable=relative-beyond-top-level
from ..config import TEST_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import SEED_VALUE, TEST_SIZE # pylint: disable=relative-beyond-top-level

def test_all():
    """
        Function to test all the preprocessing methods.
    """
    assert test_machine_learning_algorithm()
    assert test_train_algorithm()
    return True

def test_machine_learning_algorithm():
    """
        Function to test the function 'machine_learning_algorithm'.
    """
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)

    assert not exists(save_path)

    # Test function with methods not in list
    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='',
        save_path = save_path,
        verbose = False)
    assert model is None
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='LogisticRegression',
        save_path = save_path,
        verbose = False)
    assert model is None
    assert not exists(save_path)

    # Test function with methods in list
    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='logistic_regression',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='naive_bayes',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='knn',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='random_forest',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='svm',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    model = machine_learning_algorithm(
        dataset_path=dataset_path,
        algorithm='perceptron',
        save_path = save_path,
        verbose = False)
    assert model is not None
    assert exists(save_path)

    return True

def test_train_algorithm():
    """
        Function to test the function 'train_algorithm'.
    """
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    dataset = pd.read_csv(dataset_path)

    train, _ = train_test_split(dataset, test_size=TEST_SIZE, random_state=SEED_VALUE)

    train_y = train["y"].to_numpy()
    train_x = train.drop(["y"], axis=1).to_numpy()

    # Test function with methods not in list
    model = train_algorithm(algorithm='', train_x=train_x, train_y=train_y)
    assert model is None

    model = train_algorithm(algorithm='LogisticRegression', train_x=train_x, train_y=train_y)
    assert model is None

    # Test function with methods in list
    model = train_algorithm(algorithm='logistic_regression', train_x=train_x, train_y=train_y)
    assert model is not None

    model = train_algorithm(algorithm='naive_bayes', train_x=train_x, train_y=train_y)
    assert model is not None

    model = train_algorithm(algorithm='knn', train_x=train_x, train_y=train_y)
    assert model is not None

    model = train_algorithm(algorithm='random_forest', train_x=train_x, train_y=train_y)
    assert model is not None

    model = train_algorithm(algorithm='svm', train_x=train_x, train_y=train_y)
    assert model is not None

    model = train_algorithm(algorithm='perceptron', train_x=train_x, train_y=train_y)
    assert model is not None

    return True
