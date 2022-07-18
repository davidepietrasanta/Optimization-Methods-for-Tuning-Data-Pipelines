"""
    Module for testing metalearner.
"""
import shutil
from os import listdir
from os.path import join, exists, isdir
import pandas as pd
import numpy as np
from src.config import TEST_FOLDER
from src.utils.metalearner import split_train_test
from src.utils.metalearner import data_preparation
from src.utils.metalearner import train_metalearner
from src.utils.metalearner import delta_or_metafeatures
from src.utils.metalearner import choose_performance_from_metafeatures
from src.exceptions import custom_value_error_test

def test_all():
    """
        Function to test all the metalearner methods.
    """
    assert test_split_train_test()
    assert test_data_preparation()
    assert test_train_metalearner()
    assert test_choose_performance_from_metafeatures()
    assert test_delta_or_metafeatures()

    assert clear_all()
    return True

def test_split_train_test():
    """
        Function to test the function 'split_train_test'.
    """

    dataframe = pd.DataFrame({'Group_ID':[1,1,1,2,2,2,3,4,5,5],
        'Item_ID':[1,2,3,4,5,6,7,8,9,10],
        'Target': [0,0,1,0,1,1,0,0,0,1]})

    all_unique = dataframe['Group_ID'].unique()

    [train, test] = split_train_test(dataframe, group_name='Group_ID')

    train_unique = train['Group_ID'].unique()
    test_unique = test['Group_ID'].unique()

    # Check none of the item in train_unique are in test_unique
    for train_value in train_unique:
        for test_value in test_unique:
            assert test_value != train_value

    # Check we didn't miss any data
    conc = np.concatenate((train_unique, test_unique))
    assert np.array_equal(all_unique.sort() ,conc.sort())

    return True

def test_data_preparation():
    """
        Function to test the function 'data_preparation'.
    """
    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Test with quotient false
    data = data_preparation(
        data_path=dataset_path,
        save_path= save_path,
        data_selection = False,
        data_preprocess = True,
        metafeatures_extraction = True,
        model_training = True,
        quotient=False,
        verbose=True)

    assert data is not None

    # Test with quotient true
    data = data_preparation(
        data_path=dataset_path,
        save_path= save_path,
        data_selection = False,
        data_preprocess = True,
        metafeatures_extraction = True,
        model_training = True,
        quotient=True,
        verbose=True)

    assert data is not None

    return True

def test_train_metalearner():
    """
        Function to test the function 'train_metalearner'.
    """
    # Test exception CustomValueError

    save_path = join(TEST_FOLDER, 'data', 'save')
    delta_path = join(save_path, "delta.csv")

    # Test function with algorithm not in list
    assert custom_value_error_test(
        train_metalearner,
        metafeatures_path = delta_path,
        algorithm='',
        save_path = save_path)

    # Test function with algorithm not in list
    assert custom_value_error_test(
        train_metalearner,
        metafeatures_path = delta_path,
        algorithm='RF',
        save_path = save_path)

    model = train_metalearner(
        metafeatures_path = delta_path,
        algorithm='random_forest',
        save_path = save_path)
    assert model is not None

    model = train_metalearner(
        metafeatures_path = delta_path,
        algorithm='knn',
        save_path = save_path)
    assert model is not None

    return True

def test_choose_performance_from_metafeatures():
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

def test_delta_or_metafeatures():
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

def clear_all():
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
                #print(sub_dir_path)
                shutil.rmtree(sub_dir_path)

    return True
