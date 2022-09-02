"""
    Meta-learner module.
    Used to:\n
        - Create the train/test data\n
        - Train the meta-learner\n
"""
from os.path import join, dirname, exists
from os import makedirs
import logging
import ast
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from src.config import DATASET_FOLDER, METAFEATURES_MODEL_FOLDER
from src.config import METAFEATURES_FOLDER, MODEL_FOLDER
from src.config import LIST_OF_PREPROCESSING, LIST_OF_ML_MODELS
from src.config import LIST_OF_ML_MODELS_FOR_METALEARNING
from src.config import SEED_VALUE, TEST_SIZE
from src.exceptions import CustomValueError
from .machine_learning_algorithms import prediction_metrics
from .dataset_selection import select_datasets
from .metafeatures_extraction import metafeatures_extraction_data
from .machine_learning_algorithms import extract_machine_learning_performances
from .preprocessing_methods import preprocess_all_datasets
from .preprocessing_methods import categorical_string_to_number,categorical_algorithm_to_number

def data_preparation(
    data_selection:bool = False,
    dataset_size:str = 'medium',
    data_path:str = DATASET_FOLDER,
    data_preprocess:bool = True,
    metafeatures_extraction:bool = True,
    model_training:bool = True,
    save_path:str = METAFEATURES_FOLDER,
    quotient:bool = True) -> pd.DataFrame:
    """
        Given a preprocessing method and data,
         it returns the data for the meta-learning training.

        :param data_selection: Decide if download the datasets or take it elsewhere.
        :param dataset_size: Used during the dataset download,
         ignored if 'data_selection' is False.
        :param data_path: Where to find the dataset if 'data_selection' is True.
        :param save_path: Where to save the data

        :return: The data for the meta-learning.
    """
    logging.info("Data collection...")

    # Create the dir if it doesn't exist
    if not exists(save_path):
        makedirs(save_path)

    merged_data = _data_collection(
        data_selection,
        dataset_size,
        data_path,
        data_preprocess,
        metafeatures_extraction,
        model_training,
        save_path)

    logging.info("Delta metafeatures...")
    delta = _delta_metafeatures(merged_data, save_path, quotient)

    return delta

def _data_collection(
    data_selection:bool = False,
    dataset_size:str = 'medium',
    data_path:str = DATASET_FOLDER,
    data_preprocess:bool = True,
    metafeatures_extraction:bool = True,
    model_training:bool = True,
    save_path:str = METAFEATURES_FOLDER) -> pd.DataFrame:
    """
        Given a preprocessing method and data, it returns all the data collected.

        :param data_selection: Decide if download the datasets or take it elsewhere.
        :param dataset_size: Used during the dataset download,
         ignored if 'data_selection' is False.
        :param data_path: Where to find the dataset if 'data_selection' is True.
        :param save_path: Where to save the data

        :return: The data for the meta-learning.
    """
    dataset_path = data_path
    metafeatures_path = join(dataset_path, 'metafeatures')
    performance_path = join(dataset_path, 'performance')

    # Dataset selection
    # Takes a long time (need internet connection)
    dataset_path = _data_selection(data_selection, dataset_size, data_path)

    # Preprocess all the datasets with all the methods.
    # Save the preprocessed data in the same folder.
    _data_preprocess(data_preprocess, dataset_path)

    # Extract the meta-features.
    # Save the meta-features in the same folder.
    _metafeatures_extraction(metafeatures_extraction, dataset_path, metafeatures_path)

    # Run the ml models for all the dataset with all preprocessing.
    # Save the meta-features in the same folder.
    _model_training(model_training, dataset_path, performance_path)

    # Merge the data
    merged_data = _merge_data_all(performance_path, metafeatures_path)

    # Save the data
    _save(merged_data, save_path)

    return merged_data

def _merge_data(performance_path:str, metafeatures_path:str) -> pd.DataFrame:
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

    expanded_metafeatures.drop(
        columns=['dataset_name', 'Unnamed: 0'],
        inplace=True,
        errors='ignore')

    performances.drop(
        columns=['index', 'Unnamed: 0'],
        inplace=True,
        errors='ignore')

    expanded_metafeatures.reset_index(inplace=True, drop=True)
    performances.reset_index(inplace=True, drop=True)

    performances = pd.concat([performances, expanded_metafeatures], axis=1)
    return performances

def _merge_data_all(performance_dir:str, metafeatures_dir:str) -> pd.DataFrame:
    """
        Used to merge data after the extraction of the performances and the metafeatures.

        :param performance_dir: Path of the performance CSV directory.
        :param metafeatures_dir: Path of the metafeatures CSV directory.

        :return: The merged data.
    """
    logging.info("Merging the data... [5/6]")

    logging.info("Merging from the non-preprocessed data")
    no_preprocessed_metafeatures = join(metafeatures_dir, 'metafeatures_data.csv')
    no_preprocessed_performances = join(performance_dir, 'performance.csv')
    merged_data = _merge_data(no_preprocessed_performances, no_preprocessed_metafeatures)

    for preprocessing in LIST_OF_PREPROCESSING:
        logging.info("Merging %s", preprocessing)

        preprocessed_metafeatures = join(metafeatures_dir, 'metafeatures_'+ preprocessing +'.csv')
        preprocessed_performances = join(performance_dir, 'performance_'+ preprocessing +'.csv')
        temp_merged_data = _merge_data(preprocessed_performances, preprocessed_metafeatures)

        # merge with the other data
        merged_data = pd.concat([merged_data, temp_merged_data], axis=0, ignore_index=True)

        # drop null
        merged_data.dropna(subset=['attr_conc.mean'], inplace=True)
        merged_data.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

    logging.info("Data Merged [5/6]")

    return merged_data

def _data_selection(data_selection:bool, dataset_size:str, data_path:str) -> str:
    """
        Function to performe the dataset selection
    """
    dataset_path = data_path

    if data_selection:
        logging.info("Downloading the data... [1/6]")

        # Download the datasets
        select_datasets(size=dataset_size, save_path=data_path)
        dataset_path = join(data_path, dataset_size)
    logging.info("Data Downloaded [1/6]")

    return dataset_path

def _data_preprocess(data_preprocess:bool, dataset_path:str) -> None:
    """
        Function to performe the data preprocessing
    """
    if data_preprocess:
        logging.info("Preprocessing the data... [2/6]")

        preprocess_all_datasets(
            datasets_path = dataset_path,
            save_path = dataset_path)

    logging.info("Data Preprocessed [2/6]")

def _metafeatures_extraction(
    metafeatures_extraction:bool,
    dataset_path:str,
    metafeatures_path:str) -> None:
    """
        Function to performe the meta-features extraction
    """
    if metafeatures_extraction:
        logging.info("Extracting the meta-features... [3/6]")

        logging.info("Extracting from the non-preprocessed dataset - 0/%s",
        str(len(LIST_OF_PREPROCESSING)))
        metafeatures_extraction_data(
            datasets_path = dataset_path,
            save_path = metafeatures_path,
            name_saved_csv = None)

        for i, preprocessing in enumerate(LIST_OF_PREPROCESSING):
            logging.info("Extracting %s - %s/%s",
            preprocessing, str(i+1), str(len(LIST_OF_PREPROCESSING)))

            data_preprocessing_path = join(dataset_path, preprocessing)
            metafeatures_extraction_data(
                datasets_path = data_preprocessing_path,
                save_path = metafeatures_path,
                name_saved_csv = 'metafeatures_'+ preprocessing +'.csv')
    logging.info("Meta-features Extracted [3/6]")

def _model_training(
    model_training:bool,
    dataset_path:str,
    performance_path:str) -> None:
    """
        Function to performe the model training
    """
    if model_training:
        logging.info("Training the models... [4/6]")

        logging.info(
            "Training with the non-preprocessed dataset (0/%s)",
            str(len(LIST_OF_PREPROCESSING)))
        extract_machine_learning_performances(
            datasets_path = dataset_path,
            save_model_path = MODEL_FOLDER,
            save_performance_path = performance_path,
            performance_file_name= 'performance.csv',
            preprocessing = 'None')

        for n_preprocessing, preprocessing in enumerate(LIST_OF_PREPROCESSING):
            logging.info(
                "Training with %s (%s/%s)",
                preprocessing, str(n_preprocessing+1), str(len(LIST_OF_PREPROCESSING)))

            data_preprocessing_path = join(dataset_path, preprocessing)
            extract_machine_learning_performances(
                datasets_path = data_preprocessing_path,
                save_model_path = MODEL_FOLDER,
                save_performance_path = performance_path,
                performance_file_name= 'performance_'+ preprocessing +'.csv',
                preprocessing = preprocessing)
    logging.info("Models Trained [4/6]")

def _save(data:pd.DataFrame, save_path:str) -> None:
    """
        Function to save the data
    """
    logging.info("Saving the data... [6/6]")

    save_dir = join(save_path, 'metafeatures.csv')
    data.to_csv(save_dir)

    logging.info("Data Saved [6/6]")

def _delta_metafeatures(
    metafeatures:pd.DataFrame,
    save_path:str = METAFEATURES_FOLDER,
    quotient:bool = True) -> pd.DataFrame:
    """
        Given the metafeatures, it returns the delta
         performance and meta-features of the dataset.

        :param metafeatures: The CSV file with all the metafeatures and
         performances collected.
         It should be the output of the 'data_collection' function.
        :param quotient: If true calculate the delta as quotient
         else as a subtraction.

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

                if len(preprocessed.index) > 0 and len(non_preprocessed) > 0:
                    preprocessed = preprocessed.iloc[0].to_numpy()
                    preprocessed[0] = ast.literal_eval(preprocessed[0])['f1_score']

                    logging.info("Delta of '%s' with '%s' and '%s'.",
                        dataset, ml_model, preprocessing)

                    delta = delta_funct(preprocessed, non_preprocessed, quotient)

                    # Add dataset_name, algorithm and preprocessing
                    info = np.array([dataset, ml_model, preprocessing])
                    delta = pd.DataFrame(np.concatenate(( info.flatten(), delta.flatten()) ).T)

                    # Add the delta to the dataframe
                    delta_df = pd.concat([delta_df, delta], axis=1)

    delta_df = delta_df.transpose()
    delta_df.set_axis(metafeatures.columns.values.tolist(), axis=1, inplace=True)
    delta_df.rename(
        columns={
            'preprocessing': 'dataset_name',
            'algorithm': 'preprocessing',
            'dataset_name': 'algorithm'},
        inplace=True)
    delta_df.to_csv(join(save_path, 'delta.csv'), index=False)
    return delta_df

def choose_performance_from_metafeatures(
    metafeatures_path:str,
    metric:str='f1_score',
    copy_name:str='new_metafeatures.csv') -> pd.DataFrame:
    """
        Create a copy of the metafeatures file but with just one performance.

        :param metafeatures_path: Path to the dataset, should be a CSV file
        generated by 'data_preparation'.
        :param performance: Performance to keep.
        :param copy_name: Name of the new file.

        :return: The new dataset.
    """
    metafeatures = pd.read_csv(metafeatures_path)
    all_performances = metafeatures['performance'].to_numpy()
    perf_metric = []
    for performance in all_performances:
        perf_metric.append( ast.literal_eval(performance) [metric] )

    metafeatures = metafeatures.drop(["Unnamed: 0"], axis=1)
    metafeatures["performance"] = perf_metric

    metafeatures_path = dirname(metafeatures_path)
    metafeatures.to_csv(join(metafeatures_path, copy_name), index=False)
    return metafeatures

def train_metalearner(
    metafeatures_path:str,
    algorithm:str ='random_forest',
    save_path:str = METAFEATURES_MODEL_FOLDER):
    """
        Given a dataset and a model it train
         the meta-learner.

        :param algorithm: A machine learning model/algorithm.
         It should be in
            ["knn",
            "random_forest",
            "gaussian_process"]
        :param metafeatures_path: Path to the dataset, should be a CSV file
        generated by 'data_preparation'.
        :param save_path: Where to save the trained model.

        :return: A trained meta-learning and its performances.
    """
    logging.info("Training meta-learner...")

    if algorithm not in LIST_OF_ML_MODELS_FOR_METALEARNING:
        raise CustomValueError(list_name='ml_models_for_metalearning', input_value=algorithm)

    metafeatures = pd.read_csv(metafeatures_path)
    metafeatures = categorical_algorithm_to_number(metafeatures)
    metafeatures = categorical_string_to_number(metafeatures)

    # Split train and test
    logging.info("Splitting train and test...")
    [train, test] = split_train_test(metafeatures, group_name='dataset_name')

    # Drop dataset_name
    train = train.drop(["dataset_name", "preprocessing"], axis=1)
    test = test.drop(["dataset_name", "preprocessing"], axis=1)

    train_y = train["performance"].to_numpy()
    train_x = train.drop(["performance"], axis=1).to_numpy()

    test_y = test["performance"].to_numpy()
    test_x = test.drop(["performance"], axis=1).to_numpy()

    # Train
    logging.info("Training...")
    ml_model = _train(algorithm, train_x, train_y)

    # Save
    logging.info("Saving the model...('%s')", algorithm)
    file_path = join(save_path, 'metalearner_' + algorithm + '.joblib')

    if not exists(save_path):
        makedirs(save_path)

    dump(ml_model, file_path)

    # Performance
    performances = prediction_metrics(ml_model, test_x, test_y, metrics = None, regression=True)
    logging.info("Performances: %s", performances)
    return [ml_model, performances]

def _train(algorithm:str, train_x, train_y):

    if algorithm not in LIST_OF_ML_MODELS_FOR_METALEARNING:
        raise CustomValueError(list_name='ml_models_for_metalearning', input_value=algorithm)

    if algorithm == 'knn':
        n_classes = len( set(train_y) )
        model = knn_regression(train_x, train_y, n_classes)
    elif algorithm == 'random_forest':
        model = random_forest_regression(train_x, train_y)
    elif algorithm == 'gaussian_process':
        model = gaussian_process(train_x, train_y)

    return model

def knn_regression(x_train, y_train, n_neighbors:int):
    """
        Given X and y return a trained K-Neighbors model.

        :param X: Input variables
        :param y: Label or Target value
        :param n_neighbors: Number of neighbors to consider

        :return: A trained model.
    """
    model = KNeighborsRegressor(
        n_neighbors = n_neighbors,
        weights = "distance"
        ).fit(x_train, y_train)
    return model

def random_forest_regression(x_train, y_train):
    """
        Given X and y return a trained Random Forest model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    model = RandomForestRegressor(
        random_state=SEED_VALUE
        ).fit(x_train, y_train)
    return model

def gaussian_process(x_train, y_train):
    """
        Given X and y return a trained Gaussian Process model.

        :param X: Input variables
        :param y: Label or Target value

        :return: A trained model.
    """
    kernel = ConstantKernel(
        1.0,
        #constant_value_bounds="fixed"
        )*RBF(
            1.0,
            #length_scale_bounds="fixed"
            )

    model = GaussianProcessRegressor(
        kernel = kernel,
        n_restarts_optimizer = 5,
        random_state=SEED_VALUE
        ).fit(x_train, y_train)
    return model

def split_train_test(
    dataframe:pd.DataFrame,
    group_name:str,
    test_size:float = TEST_SIZE,
    random_state:int = SEED_VALUE):
    """
        Split a dataframe into train and test keeping
         in the same split items with the same group_name.

        :param dataframe: Dataframe to split.
        :param group_name: Label of the dataframe you want
         to consider to keep the same items in the same split.
        :param test_size: Represent the proportion
        of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied
         to the data before applying the split.

        :return: Return train and test [train, test]
    """

    splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state)
    split = splitter.split(dataframe, groups=dataframe[group_name])
    train_inds, test_inds = next(split)

    train = dataframe.iloc[train_inds]
    test = dataframe.iloc[test_inds]

    return [train, test]

def delta_or_metafeatures(
    delta_path:str,
    metafeatures_path:str,
    algorithm:str = 'random_forest') -> bool:
    """
        Check if it's better to use delta_metafeatures or metafeatures.
        True if delta_metafeatures is better than metafeatures, else False.

        :param delta_path: Path to delta_metafeatures CSV file (delta.csv).
        :param metafeatures_path: Path to metafeatures CSV file (metafeatures.csv).
        :param algorithm: A machine learning model/algorithm.

        :return: True if delta_metafeatures is better than metafeatures, else False.

    """
    if algorithm not in LIST_OF_ML_MODELS_FOR_METALEARNING:
        raise CustomValueError(list_name='ml_models_for_metalearning', input_value=algorithm)

    [_, delta_performances ] = train_metalearner(
        metafeatures_path = delta_path,
        algorithm=algorithm)

    choose_performance_from_metafeatures(
        metafeatures_path = metafeatures_path,
        metric='f1_score',
        copy_name='new_metafeatures.csv')

    new_metafeatures_path = join( dirname(metafeatures_path), "new_metafeatures.csv")

    [_, meta_performances ] = train_metalearner(
        metafeatures_path = new_metafeatures_path,
        algorithm=algorithm)

    d_or_m = delta_performances['mse'] < meta_performances['mse']

    logging.info("The delta mse is: %s", str(delta_performances['mse']) )
    logging.info("The metafeatures mse is: %s", str(meta_performances['mse']) )

    winner = "'metafeatures'"
    if d_or_m:
        winner = "'delta'"

    logging.info('The performances of %s are better.', winner)

    return d_or_m

def delta_funct(
    preprocessed:np.array,
    non_preprocessed:np.array,
    quotient:bool=True) -> np.array:
    """
        Function to calculate the delta.
        Avoid to have error with quotient.

        :param preprocessed, preprocessed np.array
        :param non_preprocessed, non preprocessed np.array
        :param quotient, If true do quotient, else subtraction

        :return: The delta between preproccessed and non_preprocessed.
    """
    if quotient:
        # Avoid sys.float_info.min or too small numbers
        # because it would lead to inf results
        epsilon = 1e-4
        delta = []

        # pylint: disable=consider-using-enumerate
        for index in range(len(non_preprocessed)):

            # To avoid denominator at zero
            if non_preprocessed[index] == 0:
                logging.debug("Denominator equal to zero (index '%d').",index)
                non_preprocessed[index] = epsilon

            # 0/0 should be 1, since there is no change
            if preprocessed[index] == 0:
                logging.debug("Nominator equal to zero (index '%d').",index)
                preprocessed[index] = epsilon

            # positive/negative should be >1, since 4/-2 is increasing
            # so 4/-2 should be (4 + 2)/2 = 6/2 = 3, since is increasing of 3 time
            if preprocessed[index] > 0 > non_preprocessed[index]:
                preprocessed[index] = abs(preprocessed[index] - non_preprocessed[index])
                non_preprocessed[index] = abs(non_preprocessed[index])

            # negative/negative should be
            # * <1, if nominator > denominator since -4/-2 is decreasing
            # * >1, if nominator < denominator since -2/-6 is increasing
            # so -4/-2 should be 2/4=1/2=0.5, since is decreasing
            # and -2/-6 should be 6/3=2, since is increasing
            if preprocessed[index] < 0 and non_preprocessed[index] < 0:
                abs_preprocessed = abs(preprocessed[index])
                preprocessed[index] = abs(non_preprocessed[index])
                non_preprocessed[index] = abs_preprocessed

            # negative/positive should be <1, since is decreasing
            # -4/2 = 2/(4+2) = 2/6 = 1/3 = 0.33
            if preprocessed[index] < 0 < non_preprocessed[index]:
                preprocessed[index] = abs(preprocessed[index] - non_preprocessed[index])
                abs_preprocessed = preprocessed[index]
                preprocessed[index] = abs(non_preprocessed[index])
                non_preprocessed[index] = abs_preprocessed

            delta.append(preprocessed[index]/non_preprocessed[index])

    else:
        delta = np.diff([preprocessed, non_preprocessed], axis=0)

    delta = np.asarray(delta)
    return delta

# To DO:
# Choose a model/algorithm
