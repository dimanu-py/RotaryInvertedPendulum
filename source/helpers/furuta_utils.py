import os

import pandas as pd
import yaml


def read_yaml_parameters(yaml_path: str = None) -> dict:
    """
    Read yaml file and load content in a dictionary.
    :param yaml_path: path to yaml file
    :return: content as a dictionary
    """
    if yaml_path is None:  # default path
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/parameters.yaml'))

    with open(yaml_path, 'r') as params_file:
        try:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters

        except Exception as e:
            print(f'Error loading yaml {e.args[1]}')


# TODO: encapsular esta funcionalidad en una clase extendible como con data_saver.py
def load_parquet(path: str, file: str) -> pd.DataFrame:
    """
    Load parquet file into a dataframe
    :param path: folder path of the file
    :param file: name fo the file to read
    :return: data as a dataframe
    """
    try:
        data = pd.read_parquet(path=f"{path}/{file}.parquet")
        return data

    except Exception as e:
        print(f'Error reading parquet file {e.args[1]}')
