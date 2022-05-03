import os
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent

CODE_FOLDER = os.path.join(ROOT_FOLDER, "src", "code")
DATASET_FOLDER = os.path.join(ROOT_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")
OUT_FOLDER = os.path.join(ROOT_FOLDER, "src", "out")
UTILS_FOLDER = os.path.join(ROOT_FOLDER, "src", "utils")

LIST_OF_DATASETS = []

LIST_OF_PREPROCESSING = []

LIST_OF_MODELS = []