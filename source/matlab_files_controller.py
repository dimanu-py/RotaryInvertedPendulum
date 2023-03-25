import os

import hdf5storage
import pandas as pd

from source.furuta_utils import read_yaml_parameters


class MatlabFilesController:

    def __init__(self):
        self.yaml_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/matlab_config.yaml'))
        configuration_params = read_yaml_parameters(yaml_path=self.yaml_config_path)
        self.mat_file_path = configuration_params['mat_file']['path']
        self.mat_data_name = configuration_params['mat_file']['data_name']
        self.columns_name = configuration_params['mat_file']['columns_name']
        self.drop_time_column = configuration_params['drop_time_column']
        save_dir = configuration_params['save_file']['path']
        save_file_name = configuration_params['save_file']['file_name']
        self.save_path = os.path.join(save_dir, save_file_name)

    def save_file_as_parquet(self) -> None:

        data = self.get_signals_data()

        data.to_parquet(path=self.save_path,
                        index=False)

    def get_signals_data(self) -> pd.DataFrame:

        matlab_data = self.load_matlab_file()
        signals_data = matlab_data.get(self.mat_data_name).transpose()

        df_signals = pd.DataFrame(data=signals_data,
                                  columns=self.columns_name)

        df_signals = df_signals if not self.drop_time_column else df_signals.drop(columns=['time'])

        return df_signals

    def load_matlab_file(self) -> dict:
        matlab_data = hdf5storage.loadmat(file_name=self.mat_file_path)
        return matlab_data
