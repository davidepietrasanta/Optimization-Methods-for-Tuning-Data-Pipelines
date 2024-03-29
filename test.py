"""
    Module for testing all.
"""
import shutil
from os.path import join, exists
from src.test.preprocessing_methods_test import test_all as test_preprocessing_methods
from src.test.metafeatures_extraction_test import test_all as test_metafeatures_extraction
from src.test.machine_learning_test import test_all as test_machine_learning
from src.test.metalearner_test import test_all as test_metalearner
from src.test.data_preparation_test import test_all as test_data_preparations
from src.test.preprocessing_improvement_test import test_all as test_preprocessing_improvement
from src.config import TEST_FOLDER

def test_all() -> bool:
    """
        Function to test all the methods.
        ATTENTION, during the tests the models of ml are overwritten, do a backup first.
    """

    clean_test()

    print("Test...")

    assert test_preprocessing_methods()
    clean_test()
    print("Test on 'preprocessing_methods' passed.")

    assert test_metafeatures_extraction()
    clean_test()
    print("Test on 'metafeatures_extraction' passed.")

    assert test_machine_learning()
    clean_test()
    print("Test on 'machine_learning' passed.")

    assert test_data_preparations()
    print("Test on 'data_preparations' passed.")
    assert test_metalearner()
    clean_test()
    print("Test on 'metalearner' passed.")

    assert test_preprocessing_improvement()
    clean_test()
    print("Test on 'preprocessing_improvement' passed.")

    print("All tests passed!")
    return True

def clean_test() -> bool:
    """
        Makes sure everything is clean for the test.
        Deletes unnecessary directories, etc..
    """
    # delete save directory, used to store temporary data
    save_path = join(TEST_FOLDER, 'data', 'save')
    if exists(save_path):
        shutil.rmtree(save_path)

    assert not exists(save_path)
    return True

if __name__ == '__main__':
    test_all()
