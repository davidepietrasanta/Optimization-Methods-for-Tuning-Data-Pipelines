# Optimization Methods for Tuning Data Pipelines

Code connected with the master's thesis "Optimization Methods for Tuning Data Pipelines" by Davide Pietrasanta.

## Pipeline map

![Pipeline map](/images/Pipeline.jpg)

## Delta

We want to be able to predict the delta performance, i.e. the difference between the performances obtained with and without preprocessing.

This will be the output of the meta-learner.

![Delta](/images/Delta.jpg)

## Mind Map

![Mind map](/images/Mind%20Map.jpg)

## Folders

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
|   └─ config.py
|── main.py
|── Tutorial.ipynb
└── test.py
```
