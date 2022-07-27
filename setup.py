"""
    Setup file
"""
from os import makedirs
from os.path import exists, join
import pathlib
from setuptools import setup

def setup_dir():
    """
        Setup the directories:
        -   Create data directory and all the
         sub-directories.
    """
    running_dir = pathlib.Path(__file__).parent.resolve()
    data = join(running_dir, "data")
    dataset = join(data, "dataset")
    metafeatures = join(data, "metafeatures")
    model = join(data, "model")

    if not exists(data):
        makedirs(data)

    if not exists(dataset):
        makedirs(dataset)

    if not exists(metafeatures):
        makedirs(metafeatures)

    if not exists(model):
        makedirs(model)

setup_dir()
setup(name='Optimization-Methods-for-Tuning-Data-Pipelines', version='1.0')
