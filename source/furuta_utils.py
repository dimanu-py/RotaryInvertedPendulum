import os

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
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
            print(e)


def save_file_as_parquet(data: pd.DataFrame, save_path: str) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path=save_path,
                        index=False)

    else:
        with open(save_path, 'wb') as file:
            table = pyarrow.Table.from_pandas(df=data)
            pq.write_table(table=table,
                           where=file)


def load_parquet(path, file):
    return pd.read_parquet(path=f"{path}/{file}.parquet")
