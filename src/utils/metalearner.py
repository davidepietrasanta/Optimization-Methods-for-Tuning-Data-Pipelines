"""
    Meta-learner module.
    Used to:\n
        - Create the train/test data\n
        - Train the meta-learner\n
"""
from os.path import join
from .dataset_selection import select_datasets # pylint: disable=relative-beyond-top-level
from .metafeatures_extraction import metafeatures_extraction_data # pylint: disable=relative-beyond-top-level
from .machine_learning_algorithms import extract_machine_learning_performances # pylint: disable=relative-beyond-top-level
from .preprocessing_methods import preprocess_all_datasets # pylint: disable=relative-beyond-top-level
from ..config import DATASET_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import METAFEATURES_FOLDER, MODEL_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import LIST_OF_PREPROCESSING # pylint: disable=relative-beyond-top-level


def data_preparation(
    data_selection = False,
    dataset_size = 'medium',
    data_path = DATASET_FOLDER,
    data_preprocess = True,
    metafeatures_extraction = True,
    model_training = True,
    save_path= METAFEATURES_FOLDER,
    verbose=False):
    """
        Given a preprocessing method and data, it returns transformed data.

        :param data_selection: Decide if download the datasets or take it elsewhere.
        :param dataset_size: Used during the dataset download,
         ignored if 'data_selection' is False.
        :param data_path: Where to find the dataset if 'data_selection' is True.
        :param save_path: Where to save the data
        :param verbose: If True more info are printed.

        :return: The data for the meta-learning.
    """
    dataset_path = data_path
    # Takes a long time
    if data_selection:
        if verbose:
            print("Downloading the data... [1/4]")

        # Download the datasets
        select_datasets(size=dataset_size, verbose=verbose, save_path=data_path)
        dataset_path = join(data_path, dataset_size)
    if verbose:
        print("Data Downloaded [1/4]")

    # Preprocess all the datasets with all the methods.
    # Save the preprocessed data in the same folder.
    if data_preprocess:
        if verbose:
            print("Preprocessing the data... [2/4]")

        preprocess_all_datasets(
            datasets_path = dataset_path,
            save_path = dataset_path,
            verbose = verbose)
    if verbose:
        print("Data Preprocessed [2/4]")

    # Extract the meta-features.
    # Save the meta-features in the same folder.
    if metafeatures_extraction:
        if verbose:
            print("Extracting the meta-features... [3/4]")
        metafeatures_path = join(dataset_path, 'metafeatures')

        if verbose:
            print("Extracting from the non-preprocessed dataset")
        metafeatures_extraction_data(
            datasets_path = dataset_path,
            save_path = metafeatures_path,
            name_saved_csv = None,
            verbose = False)

        for preprocessing in LIST_OF_PREPROCESSING:
            if verbose:
                print("Extracting " + preprocessing)

            data_preprocessing_path = join(dataset_path, preprocessing)
            metafeatures_extraction_data(
                datasets_path = data_preprocessing_path,
                save_path = metafeatures_path,
                name_saved_csv = 'metafeatures_'+ preprocessing +'.csv',
                verbose = False)
    if verbose:
        print("Meta-features Extracted [3/4]")

    # Run the ml models for all the dataset with all preprocessing.
    if model_training:
        if verbose:
            print("Training the models... [4/4]")

        if verbose:
            print("Training with the non-preprocessed dataset")
        extract_machine_learning_performances(
            datasets_path = dataset_path,
            save_model_path = MODEL_FOLDER,
            save_performance_path = METAFEATURES_FOLDER,
            performance_file_name= 'performance.csv',
            preprocessing = 'None',
            verbose = False)

        for preprocessing in LIST_OF_PREPROCESSING:
            if verbose:
                print("Training with " + preprocessing)

            data_preprocessing_path = join(dataset_path, preprocessing)
            extract_machine_learning_performances(
                datasets_path = data_preprocessing_path,
                save_model_path = MODEL_FOLDER,
                save_performance_path = METAFEATURES_FOLDER,
                performance_file_name= 'performance_'+ preprocessing +'.csv',
                preprocessing = preprocessing,
                verbose = False)
    if verbose:
        print("Models Trained [4/4]")

    # TO DO:
    # Merge the data (performance and metafeatures)
    # Split data into train and test (?)
    # Save the data
    print(save_path)
