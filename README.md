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
├── src                         # Actual code
│   ├─ test                     # Test code
│   ├─ utils                    # General utility code
│   ├─ out                      # Store eventual data
|   └─ config.py                # Common knowledge
|── main.py
|── requirements.txt
|── setup.py
|── test.py                     # To test all
└── Tutorial.ipynb              # Simple notebook tutorial
```

## Install

Go in the `/PATH_TO_PROJECT/Optimization-Methods-for-Tuning-Data-Pipelines/` and run:

```console
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Execution

Run with

```python3
python3 main.py
```

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

## Mind Map

From the pre-processed and raw data, metafeatures are extracted. A ML-model is then executed on both in order to collect the performances. Delta between performances and metafeatures is calculated so that we can train the meta-learner.

![Mind map](/images/Mind%20Map.jpg)
