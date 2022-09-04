"""
    Module for testing metalearner.
"""
import shutil
from os import listdir
from os.path import join, exists, isdir
import numpy as np
from src.config import TEST_FOLDER
from src.utils.data_preparation import data_preparation
from src.utils.data_preparation import delta_or_metafeatures
from src.utils.data_preparation import choose_performance_from_metafeatures
from src.utils.data_preparation import delta_funct
from src.utils.metalearner import train_metalearner
from src.exceptions import custom_value_error_test

def test_all() -> bool:
    """
        Function to test all the metalearner methods.
    """
    assert test_data_preparation()
    assert test_choose_performance_from_metafeatures()
    assert test_delta_or_metafeatures()
    assert test_delta_funct()

    # Don't do it
    #assert clear_all()
    return True

def test_data_preparation() -> bool:
    """
        Function to test the function 'data_preparation'.
    """
    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Test with quotient false
    data = data_preparation(
        data_path=dataset_path,
        save_path= save_path,
        quotient=False)

    assert data is not None

    # Test with quotient true
    data = data_preparation(
        data_path=dataset_path,
        save_path= save_path,
        quotient=True)

    assert data is not None

    return True

def test_choose_performance_from_metafeatures() -> bool:
    """
        Function to test the function 'choose_performance_from_metafeatures'.
    """
    save_path = join(TEST_FOLDER, 'data', 'save')
    metafeatures_path = join(save_path, "metafeatures.csv")

    choose_performance_from_metafeatures(
        metafeatures_path = metafeatures_path,
        metric='f1_score',
        copy_name='new_metafeatures.csv')

    new_metafeatures_path = join(save_path, "new_metafeatures.csv")

    train_metalearner(
        metafeatures_path = new_metafeatures_path,
        algorithm='random_forest',
        save_path=save_path)

    return True

def test_delta_or_metafeatures() -> bool:
    """
        Function to test the function 'delta_or_metafeatures'.
    """
    save_path = join(TEST_FOLDER, 'data', 'save')

    delta_path = join(save_path, "delta.csv")
    metafeatures_path = join(save_path, "metafeatures.csv")

    # Test function with algorithm not in list
    assert custom_value_error_test(
        delta_or_metafeatures,
        delta_path=delta_path,
        metafeatures_path=metafeatures_path,
        algorithm='')

    # Test function with algorithm not in list
    assert custom_value_error_test(
        delta_or_metafeatures,
        delta_path=delta_path,
        metafeatures_path=metafeatures_path,
        algorithm='RF')

    choice = delta_or_metafeatures(
        delta_path=delta_path,
        metafeatures_path=metafeatures_path,
        algorithm='random_forest')

    assert choice

    choice = delta_or_metafeatures(
        delta_path=delta_path,
        metafeatures_path=metafeatures_path,
        algorithm='knn')

    assert choice

    return True

def test_delta_funct() -> bool:
    """
        Function to test the function 'delta_funct'.
    """

    #quotient = False
    num = np.array([1, 4, 8])
    denom = np.array([0, 6, -2])

    delta = delta_funct(num, denom, quotient=False)
    solution = np.array([1, -2, 10])
    assert np.array_equal(delta, solution)

    # quotient = True
    # test positive/negative
    delta = delta_funct(num, denom, quotient=True)

    assert delta[0] > 0
    assert delta[1] == 4/6
    assert delta[2] == (8+2)/2

    # quotient = True
    # test 0/0, negative/positive and negative/negative
    num = np.array([0, -4, -4, -2])
    denom = np.array([0, 2, -2, -6])
    delta = delta_funct(num, denom, quotient=True)
    solution = np.array([1, 2/(4+2), 2/4, 6/2])
    assert np.array_equal(delta, solution)

    return True

def clear_all() -> bool:
    """
        Makes sure everything is clean for the test.
        Deletes unnecessary directories, etc..
    """
    # delete all dir into data_path
    data_path = join(TEST_FOLDER, 'data')
    if exists(data_path):
        for sub_dir in listdir(data_path):
            sub_dir_path = join(data_path, sub_dir)
            if isdir(sub_dir_path):
                shutil.rmtree(sub_dir_path)

    return True
