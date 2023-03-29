import os

import pandas as pd
import yaml


def read_yaml_parameters(yaml_path: str = None) -> dict:

    if yaml_path is None:
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/parameters.yaml'))

    with open(yaml_path, 'r') as params_file:
        try:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters

        except Exception as e:
            print(f'Error loading yaml {e.args[1]}')


def save_file_as_parquet(data: pd.DataFrame, save_dir: str, save_file_name: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    full_path = os.path.join(save_dir, save_file_name)

    try:
        data.to_parquet(path=full_path,
                        index=False)

    except Exception as e:
        print(f'Error saving data as parquet {e.args[1]}')


def load_parquet(path, file):
    try:
        data = pd.read_parquet(path=f"{path}/{file}.parquet")
        return data

    except Exception as e:
        print(f'Error reading parquet file {e.args[1]}')
