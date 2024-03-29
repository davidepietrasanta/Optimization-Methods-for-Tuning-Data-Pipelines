"""
    Module dedicated to the custom exceptions.
"""
import logging
from src.config import LIST_OF_ML_MODELS, LIST_OF_ML_MODELS_FOR_METALEARNING
from src.config import LIST_OF_PREPROCESSING

class CustomValueError(ValueError):
    """
        Exception to handle the insertion of values that
         are not allowed or not in list.
    """
    def __init__(self, list_name, input_value):

        ValueError.__init__(self)

        if list_name.lower() == "ml_models":
            self.list = LIST_OF_ML_MODELS
        elif list_name.lower() == "ml_models_for_metalearning":
            self.list = LIST_OF_ML_MODELS_FOR_METALEARNING
        elif list_name.lower() == "preprocessing":
            self.list = LIST_OF_PREPROCESSING
        else:
            self.list = []

        self.message = "Your input value '" +  str(input_value) + "' is not valid! "
        self.message += "The entered value should be one of the following: "
        self.message += ", ".join(self.list)

    def __str__(self) -> str:
        return self.message

def custom_value_error_test(function, *args, **kwargs) -> bool:
    """
        Test if 'CustomValueError' exception is raised.
    """
    flag = False
    try:
        function(*args, **kwargs)
    except CustomValueError:
        flag = True
    assert flag
    return True

def exception_logging(msg:str) -> None:
    """
        In case of exception it show the msg
         both in the logging file and to the user.
    """
    logging.debug("Exception occurred", exc_info=True)
    logging.info(msg)
