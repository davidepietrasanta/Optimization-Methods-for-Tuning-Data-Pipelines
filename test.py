"""
    Module for testing all.
"""
import shutil
from os.path import join, exists
from src.test.preprocessing_methods_test import test_all as test_preprocessing_methods
from src.test.metafeatures_extraction_test import test_all as test_metafeatures_extraction
from src.test.machine_learning_test import test_all as test_machine_learning
from src.test.metalearner_test import test_all as test_metalearner
from src.config import TEST_FOLDER

def test_all(verbose=True):
    """
        Function to test all the methods.
    """

    clean_test()

    if verbose:
        print("Test...")

    #assert test_preprocessing_methods()
    clean_test()
    if verbose:
        print("Test on 'preprocessing_methods' passed.")

    #assert test_metafeatures_extraction()
    clean_test()
    if verbose:
        print("Test on 'metafeatures_extraction' passed.")

    #assert test_machine_learning()
    clean_test()
    if verbose:
        print("Test on 'machine_learning' passed.")

    assert test_metalearner()
    clean_test()
    if verbose:
        print("Test on 'metalearner' passed.")

    if verbose:
        print("All tests passed!")

    return True

def clean_test():
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
    VERBOSE = True
    #test_all(verbose=VERBOSE)
    from src.test.metalearner_test import test_data_preparation
    test_data_preparation()

# TO DO:
# Save path in test (make sure everything is saved in save_path)
