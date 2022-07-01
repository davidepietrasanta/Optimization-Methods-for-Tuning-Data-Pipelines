"""
    Module for testing preprocessing methods.
"""
import shutil
from os import listdir
from os.path import join, exists, isfile
import pandas as pd


from ..utils.preprocessing_methods import preprocess_all_datasets # pylint: disable=relative-beyond-top-level
from ..utils.preprocessing_methods import preprocessing # pylint: disable=relative-beyond-top-level
from ..utils.preprocessing_methods import categorical_string_to_number # pylint: disable=relative-beyond-top-level
from ..config import TEST_FOLDER # pylint: disable=relative-beyond-top-level
from ..exceptions import CustomValueError # pylint: disable=relative-beyond-top-level

def test_all():
    """
        Function to test all the preprocessing methods.
    """
    assert test_preprocess_all_datasets()
    assert test_preprocessing()
    assert test_categorical_string_to_number()
    return True

def test_preprocess_all_datasets():
    """
        Function to test the function 'preprocess_all_datasets'.
    """

    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)

    assert not exists(save_path)

    list_datasets_processed = preprocess_all_datasets(
        datasets_path=dataset_path,
        save_path= save_path,
        verbose=False)
    list_datasets = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    # Sorting both the lists, just to be sure
    list_datasets.sort()
    list_datasets_processed.sort()

    assert list_datasets == list_datasets_processed
    assert exists(save_path)

    return True

def test_preprocessing():
    """
        Function to test the function 'preprocessing'.
    """
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    dataset = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset)

    y_data = dataset["y"].to_numpy()
    x_data  = dataset.drop(["y"], axis=1).to_numpy()

    # Test function with methods not in list
    flag = False
    try:
        new_data = preprocessing(method='', x_data=x_data, y_data=y_data)
    except CustomValueError:
        flag = True
    assert flag

    flag = False
    try:
        new_data = preprocessing(method='MinMaxScaler', x_data=x_data, y_data=y_data)
    except CustomValueError:
        flag = True
    assert flag

    # Test function with methods in list
    new_data = preprocessing(method='min_max_scaler', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='standard_scaler', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='select_percentile', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='pca', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='fast_ica', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='feature_agglomeration', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='polynomial_features', x_data=x_data, y_data=y_data)
    assert new_data is not None

    new_data = preprocessing(method='radial_basis_function_sampler', x_data=x_data, y_data=y_data)
    assert new_data is not None

    return True

def test_categorical_string_to_number():
    """
        Function to test the function 'categorical_string_to_number'.
    """

    # Test if it works on  dataset with categorical string
    dataset_path = join(TEST_FOLDER, 'data', 'analcatdata_dmft.csv')
    dataset_with_cat  = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset_with_cat)
    assert not dataset.equals( dataset_with_cat )

    # Test if it works on  dataset without categorical string
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    dataset_without_cat  = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset_without_cat)
    assert dataset.equals( dataset_without_cat )

    return True
