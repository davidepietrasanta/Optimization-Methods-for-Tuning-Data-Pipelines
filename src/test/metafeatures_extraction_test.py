"""
    Module for testing metafeatures extraction.
"""
import shutil
from os.path import join, exists
from ..utils.metafeatures_extraction import metafeatures_extraction # pylint: disable=relative-beyond-top-level
from ..utils.metafeatures_extraction import metafeature # pylint: disable=relative-beyond-top-level
from ..config import TEST_FOLDER # pylint: disable=relative-beyond-top-level

def test_all():
    """
        Function to test all the preprocessing methods.
    """
    assert test_metafeatures_extraction()
    assert test_metafeature()
    return True

def test_metafeatures_extraction():
    """
        Function to test the function 'metafeatures_extraction'.
    """

    dataset_path = join(TEST_FOLDER, 'data')
    save_path = join(TEST_FOLDER, 'data', 'save')

    # Delete save directory and all its files
    if exists(save_path):
        shutil.rmtree(save_path)

    assert not exists(save_path)

    metafeatures = metafeatures_extraction(
        datasets_path=dataset_path,
        save_path= save_path,
        name_saved_csv = None,
        verbose=False)
        
    assert exists(save_path)
    assert exists(join(save_path, "metafeatures.csv"))
    assert metafeatures is not None

    return True

def test_metafeature():
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

    assert metafeatures_extracted['intrinsic_dim.global']
    assert metafeatures_extracted['intrinsic_dim.local.mean']

    return True
