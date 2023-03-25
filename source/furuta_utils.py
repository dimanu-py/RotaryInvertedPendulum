import os
import pyarrow
import pandas as pd
import yaml
import pyarrow.parquet as pq


def read_yaml_parameters(yaml_path=None) -> dict:

    if yaml_path is None:
        raise TypeError('yaml path must not be None')

    with open(yaml_path, 'r') as params_file:
        try:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters

        except Exception as e:
            print(e)


def save_file_as_parquet(data: pd.DataFrame, save_path: str, file_name: str) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path=f"{save_path}/{file_name}.parquet",
                        index=False)

    else:
        with open(f"{save_path}/{file_name}.parquet", 'wb') as file:
            table = pyarrow.Table.from_pandas(df=data)
            pq.write_table(table=table,
                           where=file)


def load_parquet(path, file):
    return pd.read_parquet(path=f"{path}/{file}.pkl")
