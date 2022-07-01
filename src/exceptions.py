"""
    Module dedicated to the custom exceptions.
"""
from .config import LIST_OF_ML_MODELS, LIST_OF_ML_MODELS_FOR_METALEARNING # pylint: disable=relative-beyond-top-level
from .config import LIST_OF_PREPROCESSING # pylint: disable=relative-beyond-top-level

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

    def __str__(self):
        return self.message
