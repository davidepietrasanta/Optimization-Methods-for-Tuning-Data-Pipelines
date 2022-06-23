"""
    Meta-learner module.
    Used to:\n
        - Create the train/test data\n
        - Train the meta-learner\n
"""
from os.path import join
import ast
import pandas as pd
import numpy as np
from .dataset_selection import select_datasets # pylint: disable=relative-beyond-top-level
from .metafeatures_extraction import metafeatures_extraction_data # pylint: disable=relative-beyond-top-level
from .machine_learning_algorithms import extract_machine_learning_performances # pylint: disable=relative-beyond-top-level
from .preprocessing_methods import preprocess_all_datasets # pylint: disable=relative-beyond-top-level
from ..config import DATASET_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import METAFEATURES_FOLDER, MODEL_FOLDER # pylint: disable=relative-beyond-top-level
from ..config import LIST_OF_PREPROCESSING, LIST_OF_ML_MODELS # pylint: disable=relative-beyond-top-level

def data_preparation( # pylint: disable=too-many-arguments
    data_selection = False,
    dataset_size = 'medium',
    data_path = DATASET_FOLDER,
    data_preprocess = True,
    metafeatures_extraction = True,
    model_training = True,
    save_path= METAFEATURES_FOLDER,
    verbose=False):
    """
        Given a preprocessing method and data,
         it returns the data for the meta-learning training.

        :param data_selection: Decide if download the datasets or take it elsewhere.
        :param dataset_size: Used during the dataset download,
         ignored if 'data_selection' is False.
        :param data_path: Where to find the dataset if 'data_selection' is True.
        :param save_path: Where to save the data
        :param verbose: If True more info are printed.

        :return: The data for the meta-learning.
    """
    if verbose:
        print("Data collection...")
    merged_data = _data_collection(
        data_selection,
        dataset_size,
        data_path,
        data_preprocess,
        metafeatures_extraction,
        model_training,
        save_path,
        verbose)

    if verbose:
        print("Delta metafeatures...")
    delta = _delta_metafeatures(merged_data, save_path, verbose)

    return delta

def _data_collection( # pylint: disable=too-many-arguments
    data_selection = False,
    dataset_size = 'medium',
    data_path = DATASET_FOLDER,
    data_preprocess = True,
    metafeatures_extraction = True,
    model_training = True,
    save_path= METAFEATURES_FOLDER,
    verbose=False):
    """
        Given a preprocessing method and data, it returns all the data collected.

        :param data_selection: Decide if download the datasets or take it elsewhere.
        :param dataset_size: Used during the dataset download,
         ignored if 'data_selection' is False.
        :param data_path: Where to find the dataset if 'data_selection' is True.
        :param save_path: Where to save the data
        :param verbose: If True more info are printed.

        :return: The data for the meta-learning.
    """
    dataset_path = data_path
    metafeatures_path = join(dataset_path, 'metafeatures')
    performance_path = join(dataset_path, 'performance')

    # Dataset selection
    # Takes a long time (need internet connection)
    dataset_path = _data_selection(data_selection, verbose, dataset_size, data_path)

    # Preprocess all the datasets with all the methods.
    # Save the preprocessed data in the same folder.
    _data_preprocess(data_preprocess, verbose, dataset_path)

    # Extract the meta-features.
    # Save the meta-features in the same folder.
    _metafeatures_extraction(metafeatures_extraction, verbose, dataset_path, metafeatures_path)

    # Run the ml models for all the dataset with all preprocessing.
    # Save the meta-features in the same folder.
    _model_training(model_training, verbose, dataset_path, performance_path)

    # Merge the data
    merged_data = _merge_data_all(performance_path, metafeatures_path, verbose)

    # Save the data
    _save(merged_data, verbose, save_path)

    return merged_data

def _merge_data(performance_path, metafeatures_path):
    """
        Used to merge data after the extraction of the performances and the metafeatures.

        :param performance_path: Path of the performance CSV file.
        :param metafeatures_path: Path of the metafeatures CSV file.

        :return: The merged data.
    """
    performances = pd.read_csv(performance_path)
    metafeatures = pd.read_csv(metafeatures_path)

    # make sure indexes pair with number of rows
    performances = performances.reset_index()
    occurrences = performances['dataset_name'].value_counts().to_dict()

    expanded_metafeatures = pd.DataFrame()
    for key in occurrences:
        meta = metafeatures.loc[metafeatures['dataset_name'] == key]
        for _ in range(occurrences[key]):
            expanded_metafeatures = pd.concat([expanded_metafeatures, meta], axis=0)

    expanded_metafeatures.drop(columns=['dataset_name', 'Unnamed: 0'], inplace=True)
    performances.drop(columns=['index', 'Unnamed: 0'], inplace=True)
    expanded_metafeatures.reset_index(inplace=True, drop=True)
    performances.reset_index(inplace=True, drop=True)

    performances = pd.concat([performances, expanded_metafeatures], axis=1)
    return performances

def _merge_data_all(performance_dir, metafeatures_dir, verbose):
    """
        Used to merge data after the extraction of the performances and the metafeatures.

        :param performance_dir: Path of the performance CSV directory.
        :param metafeatures_dir: Path of the metafeatures CSV directory.

        :return: The merged data.
    """
    if verbose:
        print("Merging the data... [5/6]")

    if verbose:
        print("Merging from the non-preprocessed data")
    no_preprocessed_metafeatures = join(metafeatures_dir, 'metafeatures_data.csv')
    no_preprocessed_performances = join(performance_dir, 'performance.csv')
    merged_data = _merge_data(no_preprocessed_performances, no_preprocessed_metafeatures)

    for preprocessing in LIST_OF_PREPROCESSING:
        if verbose:
            print("Merging " + preprocessing)

        preprocessed_metafeatures = join(metafeatures_dir, 'metafeatures_'+ preprocessing +'.csv')
        preprocessed_performances = join(performance_dir, 'performance_'+ preprocessing +'.csv')
        temp_merged_data = _merge_data(preprocessed_performances, preprocessed_metafeatures)

        # merge with the other data
        merged_data = pd.concat([merged_data, temp_merged_data], axis=0, ignore_index=True)

        # drop null
        merged_data.dropna(subset=['attr_conc.mean'], inplace=True)
        merged_data.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

    if verbose:
        print("Data Merged [5/6]")

    return merged_data

def _data_selection(data_selection, verbose, dataset_size, data_path):
    """
        Function to performe the dataset selection
    """
    dataset_path = data_path

    if data_selection:
        if verbose:
            print("Downloading the data... [1/6]")

        # Download the datasets
        select_datasets(size=dataset_size, verbose=verbose, save_path=data_path)
        dataset_path = join(data_path, dataset_size)
    if verbose:
        print("Data Downloaded [1/6]")

    return dataset_path

def _data_preprocess(data_preprocess, verbose, dataset_path):
    """
        Function to performe the data preprocessing
    """
    if data_preprocess:
        if verbose:
            print("Preprocessing the data... [2/6]")

        preprocess_all_datasets(
            datasets_path = dataset_path,
            save_path = dataset_path,
            verbose = verbose)

    if verbose:
        print("Data Preprocessed [2/6]")

def _metafeatures_extraction(metafeatures_extraction, verbose, dataset_path, metafeatures_path):
    """
        Function to performe the meta-features extraction
    """
    if metafeatures_extraction:
        if verbose:
            print("Extracting the meta-features... [3/6]")

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
        print("Meta-features Extracted [3/6]")

def _model_training(model_training, verbose, dataset_path, performance_path):
    """
        Function to performe the model training
    """
    if model_training:
        if verbose:
            print("Training the models... [4/6]")

        if verbose:
            print("Training with the non-preprocessed dataset")
        extract_machine_learning_performances(
            datasets_path = dataset_path,
            save_model_path = MODEL_FOLDER,
            save_performance_path = performance_path,
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
                save_performance_path = performance_path,
                performance_file_name= 'performance_'+ preprocessing +'.csv',
                preprocessing = preprocessing,
                verbose = False)
    if verbose:
        print("Models Trained [4/6]")

def _save(data, verbose, save_path):
    """
        Function to save the data
    """
    if verbose:
        print("Saving the data... [6/6]")

    save_dir = join(save_path, 'metafeatures.csv')
    data.to_csv(save_dir)

    if verbose:
        print("Data Saved [6/6]")

def _delta_metafeatures(metafeatures, save_path= METAFEATURES_FOLDER, verbose=False):
    """
        Given the metafeatures, it returns the delta
         performance and meta-features of the dataset.

        :param metafeatures: The CSV file with all the metafeatures and
         performances collected.
         It should be the output of the 'data_collection' function.

        :return: The delta data for the meta-learning.
    """
    delta_df = pd.DataFrame()

    dataset_list = metafeatures['dataset_name'].unique()
    for dataset in dataset_list:
        for ml_model in LIST_OF_ML_MODELS:
            temp = metafeatures.loc[
                ( metafeatures['dataset_name'] == dataset ) &
                ( metafeatures['algorithm'] == ml_model) ]

            preprocessing_list = temp['preprocessing'].unique()
            preprocessing_list = np.delete(
                preprocessing_list,
                np.where(preprocessing_list == ['None']))

            non_preprocessed = temp.loc[
                    ( temp['dataset_name'] == dataset ) &
                    ( temp['algorithm'] == ml_model) &
                    ( temp['preprocessing'] == 'None') ]

            non_preprocessed.drop(columns=[
                'dataset_name',
                'algorithm',
                'preprocessing'], inplace=True)

            if len(non_preprocessed.index) > 0:
                non_preprocessed = non_preprocessed.iloc[0].to_numpy()
                non_preprocessed[0] = ast.literal_eval(non_preprocessed[0])['f1_score']

            for preprocessing in preprocessing_list:

                preprocessed = temp.loc[
                    ( temp['dataset_name'] == dataset ) &
                    ( temp['algorithm'] == ml_model) &
                    ( temp['preprocessing'] == preprocessing) ]

                preprocessed.drop(columns=[
                    'dataset_name',
                    'algorithm',
                    'preprocessing'], inplace=True)

                if len(preprocessed.index) > 0:
                    preprocessed = preprocessed.iloc[0].to_numpy()
                    preprocessed[0] = ast.literal_eval(preprocessed[0])['f1_score']
                    if len(non_preprocessed) > 0:
                        if verbose:
                            print("Delta of '"+ dataset +
                            "' with '"+ ml_model +
                            "' and '"+ preprocessing +"'.")

                        delta = np.diff([preprocessed, non_preprocessed], axis=0)
                        # Add dataset_name, algorithm and preprocessing
                        info = np.array([dataset, ml_model, preprocessing])
                        delta = pd.DataFrame(np.concatenate(( info.flatten(), delta.flatten()) ).T)

                        # Add the delta to the dataframe
                        delta_df = pd.concat([delta_df, delta], axis=1)

    delta_df = delta_df.transpose()
    delta_df.set_axis(metafeatures.columns.values.tolist(), axis=1, inplace=True)
    delta_df.rename(
        columns={'preprocessing': 'performance', 'performance': 'preprocessing'},
        inplace=True)
    delta_df.to_csv(join(save_path, 'delta.csv'), index=False)
    return delta_df


# To DO:
# Train the meta-learner
# See if it's better to use delta_metafeatures or metafeatures
# Split train and test
# Choose a model
