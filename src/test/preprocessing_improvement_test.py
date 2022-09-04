"""
    Module for testing preprocessing_improvement.
"""
from os.path import join
from src.utils.preprocessing_improvement import predicted_improvement
from src.utils.preprocessing_improvement import ml_model_to_categorical
from src.config import TEST_FOLDER, METAFEATURES_MODEL_FOLDER
from src.exceptions import custom_value_error_test


def test_all() -> bool:
    """
        Function to test all the metalearner methods.
    """
    assert test_predicted_improvement()
    assert test_ml_model_to_categorical()

    return True

def test_predicted_improvement() -> bool:
    """
        Function to test the function 'predicted_improvement'.
    """
    test_dataset_path = join(TEST_FOLDER, 'data')
    new_dataset = join(test_dataset_path, 'kc1.csv')

    prediction = predicted_improvement(
        dataset_path= new_dataset,
        preprocessing = 'pca',
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_gaussian_process.joblib')
    )
    assert prediction is not None

    prediction = predicted_improvement(
        dataset_path= new_dataset,
        preprocessing_path = join(test_dataset_path, 'pca', 'kc1.csv'),
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_random_forest.joblib')
    )
    assert prediction is not None

    prediction = predicted_improvement(
        dataset_path= new_dataset,
        preprocessing_path = join(test_dataset_path, 'pca', 'kc1.csv'),
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_knn.joblib')
    )
    assert prediction is not None

    assert custom_value_error_test(
        predicted_improvement,
        dataset_path= new_dataset,
        preprocessing = 'pc',
        algorithm = 'svm',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_knn.joblib')
    )

    assert custom_value_error_test(
        predicted_improvement,
        dataset_path= new_dataset,
        preprocessing = 'pca',
        algorithm = 'sv',
        metalearner_path = join(METAFEATURES_MODEL_FOLDER, 'metalearner_knn.joblib')
    )

    return True

def test_ml_model_to_categorical() -> bool:
    """
        Function to test the function 'ml_model_to_categorical'.
    """
    assert ml_model_to_categorical("svm") == 5
    assert ml_model_to_categorical("random_forest") == 4

    assert custom_value_error_test(
        ml_model_to_categorical,
        ml_model_name="random_forestt")

    assert custom_value_error_test(
        ml_model_to_categorical,
        ml_model_name="random_fores")

    return True
