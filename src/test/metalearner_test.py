"""
    Module for testing metalearner.
"""
import shutil
from os import listdir
from os.path import join, exists, isdir
import pandas as pd
import numpy as np
from ..config import TEST_FOLDER # pylint: disable=relative-beyond-top-level
from ..utils.metalearner import split_train_test # pylint: disable=relative-beyond-top-level
from ..utils.metalearner import data_preparation # pylint: disable=relative-beyond-top-level
from ..utils.metalearner import train_metalearner # pylint: disable=relative-beyond-top-level
from ..utils.metalearner import delta_or_metafeatures # pylint: disable=relative-beyond-top-level
from ..utils.metalearner import choose_performance_from_metafeatures # pylint: disable=relative-beyond-top-level
from ..exceptions import CustomValueError # pylint: disable=relative-beyond-top-level

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
    flag = False
    try:
        _ = train_metalearner(
        metafeatures_path = delta_path,
        algorithm='',
        save_path = save_path)
    except CustomValueError:
        flag = True
    assert flag

    # Test function with algorithm not in list
    flag = False
    try:
        _ = train_metalearner(
        metafeatures_path = delta_path,
        algorithm='RF',
        save_path = save_path)
    except CustomValueError:
        flag = True
    assert flag

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
    flag = False
    try:
        delta_or_metafeatures(
            delta_path=delta_path,
            metafeatures_path=metafeatures_path,
            algorithm='')
    except CustomValueError:
        flag = True
    assert flag

    # Test function with algorithm not in list
    flag = False
    try:
        delta_or_metafeatures(
            delta_path=delta_path,
            metafeatures_path=metafeatures_path,
            algorithm='RF')
    except CustomValueError:
        flag = True
    assert flag

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
