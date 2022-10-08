# Optimization Methods for Tuning Data Pipelines

Code connected with the master's thesis "Optimization Methods for Tuning Data Pipelines" by Davide Pietrasanta.

## Folders

Check if you have something like this

```text
lib
├── data                        # Store data
│   ├─ dataset                  # Store datasets
│   ├─ metafeatures             # Store metafeatures
│   └─ model                    # Store trained ML models
├── images                      # Images for presentations, README, etc.
├── other                       # Script or Notebook related to the thesis or to the plots
├── src                         # Actual code
│   ├─ test                     # Test code
│   ├─ utils                    # General utility code
│   ├─ exceptions.py            # To handle custom exceptions
|   └─ config.py                # Common knowledge for the project
|── main.py
|── requirements.txt
|── setup.py
|── test.py                     # To test all
└── Tutorial.ipynb              # Simple notebook tutorial
```

## Install & Setup

Go in the `/PATH_TO_PROJECT/Optimization-Methods-for-Tuning-Data-Pipelines/` and run:

```console
virtualenv venv
source venv/local/bin/activate # or source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

If running on a pc with high number of possible jobs
it's suggested to run the following command to avoid
`BLAS : Program is Terminated. Because you tried to allocate too many memory regions` error.

```console
export OMP_NUM_THREADS=1
```

<!---
    export OPENBLAS_NUM_THREADS=1
    export GOTO_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    
    The most important is "export OMP_NUM_THREADS=1"
    the other are just precautions
--->

## Execution

Run with

```python3
python3 main.py
```

To better understand how to use the framework you can consult the `Tutorial.ipynb` file.

## Test

Test all with

```python3
python3 test.py
```

## Code quality

To check the code quality with Pylint

```console
pylint $(git ls-files '*.py') > code-quality.txt
```

## Pipeline map

We want to give the users the opportunity to test their ideas or let the machine do it.

![Pipeline map](/images/Pipeline.jpg)

## Delta

We want to be able to predict the delta performance, i.e. the difference between the performances obtained with and without preprocessing.

This will be the output of the meta-learner.

![Delta](/images/Delta.jpg)

## Training Dataset Map

A simple scheme on how the dataset used for the training of the Meta-learner is created.

![Dataset map](/images/Dataset%20Map.jpg)

## Mind Map

From the pre-processed and raw data, metafeatures are extracted. A ML-model is then executed on both in order to collect the performances. Delta between performances and metafeatures is calculated so that we can train the meta-learner.

![Mind map](/images/Mind%20Map.jpg)

## Dependency

File dependency of the project.

![Dependency](/images/architecture.jpg)
