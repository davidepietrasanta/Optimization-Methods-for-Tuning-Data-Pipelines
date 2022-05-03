import openml
import pandas as pd
from openml.datasets import get_dataset
from ..config import DATASET_FOLDER

def select_datasets(save_path = DATASET_FOLDER):

    openml_list = openml.datasets.list_datasets()  # returns a dict

    # Show a nice table with some key data properties
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

    print(f"First 10 of {len(datalist)} datasets...")
    datalist.head(n=10)

    # The same can be done with lesser lines of code
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    print( openml_df.head(n=10) )
    
    # ## Exercise 1
    # 
    # * Find datasets with more than 10000 examples.
    # * Find a dataset called 'eeg_eye_state'.
    # * Find all datasets with more than 50 classes.
    # 
    # 

    print( datalist[datalist.NumberOfInstances > 10000].sort_values(["NumberOfInstances"]).head(n=20) )
    ""
    print( datalist.query('name == "eeg-eye-state"') )
    ""
    print( datalist.query("NumberOfClasses > 50") )

    # ### Download datasets
    # 
    # 

    # This is done based on the dataset ID.
    dataset = openml.datasets.get_dataset(1471)

    # Print a summary
    print(
        f"This is dataset '{dataset.name}', the target feature is "
        f"'{dataset.default_target_attribute}'"
    )
    print(f"URL: {dataset.url}")
    print(dataset.description[:500])

    # Get the actual data.
    # 
    # The dataset can be returned in 3 possible formats: as a NumPy array, a SciPy
    # sparse matrix, or as a Pandas DataFrame. The format is
    # controlled with the parameter ``dataset_format`` which can be either 'array'
    # (default) or 'dataframe'. Let's first build our dataset from a NumPy array
    # and manually create a dataframe.
    # 
    # 

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    eeg = pd.DataFrame(X, columns=attribute_names)
    eeg["class"] = y
    print(eeg[:10])

    # Instead of manually creating the dataframe, you can already request a
    # dataframe with the correct dtypes.
    # 
    # 

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    print(X.head())
    print(X.info())

    #save the dataframe
    path = save_path + '/test_dataset.csv'
    print(path)
    X.to_csv(path, index=False) 

    # Sometimes you only need access to a dataset's metadata.
    # In those cases, you can download the dataset without downloading the
    # data file. The dataset object can be used as normal.
    # Whenever you use any functionality that requires the data,
    # such as `get_data`, the data will be downloaded.
    # 
    # 

    dataset = openml.datasets.get_dataset(1471, download_data=False)




