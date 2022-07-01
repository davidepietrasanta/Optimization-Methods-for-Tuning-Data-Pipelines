"""
    Module for testing metalearner.
"""
import pandas as pd
import numpy as np
from ..utils.metalearner import split_train_test # pylint: disable=relative-beyond-top-level

def test_all():
    """
        Function to test all the metalearner methods.
    """
    assert test_split_train_test()
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

# TO DO:
# Add more test
# Test exception CustomValueError
