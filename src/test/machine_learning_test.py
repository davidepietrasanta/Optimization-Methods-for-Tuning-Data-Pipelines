"""
    Module for testing preprocessing methods.
"""
import shutil
from os.path import join, exists
from sklearn.model_selection import train_test_split
import pandas as pd
from ..utils.machine_learning_algorithms import machine_learning_algorithm # pylint: disable=relative-beyond-top-level
from ..utils.machine_learning_algorithms import train_algorithm # pylint: disable=relative-beyond-top-level
from ..utils.machine_learning_algorithms import extract_machine_learning_performances # pylint: disable=relative-beyond-top-level
from ..config import TEST_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import SEED_VALUE, TEST_SIZE # pylint: disable=relative-beyond-top-level
from ..exceptions import CustomValueError # pylint: disable=relative-beyond-top-level

def test_all():
    """
        Function to test all the preprocessing methods.
    """
    assert test_extract_machine_learning_performances()
    assert test_machine_learning_algorithm()
    assert test_train_algorithm()
    return True

def test_extract_machine_learning_performances():
    """
        Function to test the function 'extract_machine_learning_performances'.
    """
    datasets_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)

    assert not exists(save_path)

    performances = extract_machine_learning_performances(
        datasets_path=datasets_path,
        save_model_path= save_path,
        save_performance_path = save_path,
        preprocessing = None,
        verbose=False)

    assert exists(save_path)
    assert exists(join(save_path, "performances.csv"))
    assert performances is not None

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

    # Test function with algorithm not in list
    flag = False
    try:
        [_, _] = machine_learning_algorithm(
            dataset_path=dataset_path,
            algorithm='',
            save_path = save_path,
            verbose = False)
    except CustomValueError:
        flag = True
    assert flag
    assert not exists(save_path)

    # Test function with algorithm not in list
    flag = False
    try:
        [_, _] = machine_learning_algorithm(
            dataset_path=dataset_path,
            algorithm='LogisticRegression',
            save_path = save_path,
            verbose = False)
    except CustomValueError:
        flag = True
    assert flag
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
    algorithm = 'logistic_regression',
    dataset_path = dataset_path,
    save_path = save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
    algorithm = 'naive_bayes',
    dataset_path = dataset_path,
    save_path = save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
        algorithm = 'knn',
        dataset_path = dataset_path,
        save_path = save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
        algorithm = 'random_forest',
        dataset_path = dataset_path,
        save_path = save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
        algorithm = 'svm',
        dataset_path = dataset_path,
        save_path = save_path)

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)
    assert not exists(save_path)

    # Test function with algorithm in list
    assert _valid_machine_learning_algorithm_test(
        algorithm = 'perceptron',
        dataset_path = dataset_path,
        save_path = save_path)

    return True

def _valid_machine_learning_algorithm_test(algorithm, dataset_path, save_path):
    """
        Test function with algorithm in list
    """
    [model, prediction] = machine_learning_algorithm(
    dataset_path=dataset_path,
    algorithm=algorithm,
    save_path = save_path,
    verbose = False)
    assert model is not None
    assert prediction is not None
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
