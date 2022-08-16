"""
    Module for testing preprocessing methods.
"""
from os import listdir
from os.path import join, exists, isfile
import pandas as pd
from src.utils.preprocessing_methods import preprocess_all_datasets
from src.utils.preprocessing_methods import preprocessing
from src.utils.preprocessing_methods import categorical_string_to_number
from src.utils.preprocessing_methods import preprocess_dataset
from src.config import TEST_FOLDER, LIST_OF_PREPROCESSING
from src.config import delete_dir
from src.exceptions import custom_value_error_test

def test_all() -> bool:
    """
        Function to test all the preprocessing methods.
    """
    assert test_preprocess_all_datasets()
    assert test_preprocessing()
    assert test_categorical_string_to_number()
    assert test_preprocess_dataset()
    return True

def test_preprocess_all_datasets() -> bool:
    """
        Function to test the function 'preprocess_all_datasets'.
    """

    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    assert delete_dir(save_path)

    list_datasets_processed = preprocess_all_datasets(
        datasets_path=dataset_path,
        save_path= save_path)
    list_datasets = sorted([f for f in listdir(dataset_path) if isfile(join(dataset_path, f))])

    # Sorting both the lists, just to be sure
    list_datasets.sort()
    list_datasets_processed.sort()

    assert list_datasets == list_datasets_processed
    assert exists(save_path)

    return True

def test_preprocessing() -> bool:
    """
        Function to test the function 'preprocessing'.
    """
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    dataset = pd.read_csv(dataset_path)
    dataset = categorical_string_to_number(dataset)

    y_data = dataset["y"].to_numpy()
    x_data  = dataset.drop(["y"], axis=1).to_numpy()

    # Test function with methods not in list
    assert custom_value_error_test(
        preprocessing,
        method='',
        x_data=x_data,
        y_data=y_data)

    # Test function with methods not in list
    assert custom_value_error_test(
        preprocessing,
        method='MinMaxScaler',
        x_data=x_data,
        y_data=y_data)

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

def test_categorical_string_to_number() -> bool:
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

def test_preprocess_dataset() -> bool:
    """
        Function to test the function 'preprocess_dataset'.
    """

    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    assert delete_dir(save_path)

    # Test function with methods not in list
    assert custom_value_error_test(
        preprocess_dataset,
        dataset_path=dataset_path,
        method='',
        save_path=save_path)

    # Test function with methods not in list
    assert custom_value_error_test(
        preprocess_dataset,
        dataset_path=dataset_path,
        method='something',
        save_path=save_path)

    for method in LIST_OF_PREPROCESSING:
        data = preprocess_dataset(dataset_path=dataset_path, method=method, save_path=save_path)
        assert data is not None
        assert exists(save_path)
        # Delete save directory and all its files
        assert delete_dir(save_path)

    return True
