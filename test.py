"""
    Module for testing all.
"""

from src.test.preprocessing_methods_test import test_all as test_preprocessing_methods
from src.test.metafeatures_extraction_test import test_all as test_metafeatures_extraction
from src.test.machine_learning_test import test_all as test_machine_learning
from src.test.metalearner_test import test_all as test_metalearner

def test_all(verbose=True):
    """
        Function to test all the methods.
    """
    if verbose:
        print("Test...")

    assert test_preprocessing_methods()
    if verbose:
        print("Test on 'preprocessing_methods' passed.")

    assert test_metafeatures_extraction()
    if verbose:
        print("Test on 'metafeatures_extraction' passed.")

    assert test_machine_learning()
    if verbose:
        print("Test on 'machine_learning' passed.")

    assert test_metalearner()
    if verbose:
        print("Test on 'metalearner' passed.")


    if verbose:
        print("All tests passed!")
    return True


if __name__ == '__main__':
    VERBOSE = True
    test_all(verbose=VERBOSE)