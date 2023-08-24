import hdf5storage
import pandas as pd
from typing import List, Dict, Any
import random

matlab_path = '../data/matlab'


def create_dataset_from_matlab_data(simulation_type: str,
                                    file_name: List[str],
                                    data_name: str,
                                    columns_name: List[str]) -> pd.DataFrame:
    dataset = pd.DataFrame(columns=columns_name)

    for matlab_file in file_name:
        matlab_data_path = f'{matlab_path}/{simulation_type}/{matlab_file}.mat'

        data_from_matlab_file = hdf5storage.loadmat(file_name=matlab_data_path)
        signals_data = data_from_matlab_file.get(data_name).transpose()
        signals_data = pd.DataFrame(signals_data,
                                    columns=columns_name)
        dataset = pd.concat([dataset, signals_data])
    return dataset


def shuffle_data_with_sliding_windows(data: pd.DataFrame,
                                      window_length: int) -> pd.DataFrame:
    number_windows = len(data) // window_length + (len(data) % window_length != 0)

    sliding_windows = []
    for i in range(number_windows):
        first_index = i * window_length
        last_index = first_index + window_length
        window = data.iloc[first_index: last_index]
        sliding_windows.append(window)

    random.shuffle(sliding_windows)
    shuffled_dataset = pd.concat(sliding_windows,
                                 ignore_index=True)
    return shuffled_dataset


def generate_datasets(matlab_configuration: Dict[str, Any],
                      dataset_configuration: Dict[str, Any]):
    raw_dataset = create_dataset_from_matlab_data(simulation_type=matlab_configuration['source'],
                                                  file_name=matlab_configuration['file_name'],
                                                  data_name=matlab_configuration['data_name'],
                                                  columns_name=matlab_configuration['columns'])

    if dataset_configuration['shuffle_data']:
        shuffled_dataset = shuffle_data_with_sliding_windows(data=raw_dataset,
                                                             window_length=dataset_configuration['window_length'])
    else:
        shuffled_dataset = None

    return raw_dataset, shuffled_dataset
