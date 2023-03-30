import os

import hdf5storage
import pandas as pd

from source.furuta_utils import read_yaml_parameters


class MatlabFilesController:

    def __init__(self):
        self.yaml_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                             '../config/data_inserter_config.yaml'))
        configuration_params = read_yaml_parameters(yaml_path=self.yaml_config_path).get('matlab_file')
        self.mat_file_path = configuration_params['path']
        self.mat_data_name = configuration_params['data_name']
        self.columns_name = configuration_params['columns_name']

    def get_signals_data(self) -> pd.DataFrame:

        matlab_data = self.load_matlab_file(matlab_file_path=self.mat_file_path)
        try:
            signals_data = matlab_data.get(self.mat_data_name).transpose()

            df_signals = pd.DataFrame(data=signals_data,
                                      columns=self.columns_name)

            return df_signals
        except Exception as e:
            print(f'Error converting matlab data to dataframe {e.args[1]}')

    @staticmethod
    def load_matlab_file(matlab_file_path) -> dict:
        try:
            matlab_data = hdf5storage.loadmat(file_name=matlab_file_path)
            return matlab_data
        except Exception as e:
            print(f'Error loading matlab file {e.args[1]}')
