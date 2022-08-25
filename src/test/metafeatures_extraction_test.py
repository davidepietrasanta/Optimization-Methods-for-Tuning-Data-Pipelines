"""
    Module for testing metafeatures extraction.
"""
from os.path import join, exists
from src.utils.metafeatures_extraction import metafeatures_extraction_data
from src.utils.metafeatures_extraction import metafeature
from src.config import TEST_FOLDER
from src.config import delete_dir

def test_all() -> bool:
    """
        Function to test all the preprocessing methods.
    """
    assert test_metafeatures_extraction_data()
    assert test_metafeature()
    return True

def test_metafeatures_extraction_data() -> bool:
    """
        Function to test the function 'metafeatures_extraction'.
    """

    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    assert delete_dir(save_path)

    metafeatures = metafeatures_extraction_data(
        datasets_path=dataset_path,
        save_path= save_path,
        name_saved_csv = None)

    assert exists(save_path)
    assert exists(join(save_path, "metafeatures_data.csv"))
    assert metafeatures is not None

    return True

def test_metafeature() -> bool:
    """
        Function to test the function 'metafeature'.
    """

    # Test with non-existent file
    dataset_path = join(TEST_FOLDER, 'data', 'adam.csv')
    metafeatures_extracted = metafeature(dataset_path)
    assert metafeatures_extracted is None

    # Test with existent file
    dataset_path = join(TEST_FOLDER, 'data', 'ada.csv')
    metafeatures_extracted = metafeature(dataset_path)
    assert metafeatures_extracted is not None

    assert metafeatures_extracted['intrinsic_dim']

    return True
