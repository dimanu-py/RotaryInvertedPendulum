import os

import hdf5storage
import pandas as pd

from source.helpers.furuta_utils import read_yaml_parameters


class MatlabFilesController:

    def __init__(self, matlab_folder: str):
        self.matlab_folder_path = matlab_folder

        configuration_params = read_yaml_parameters().get('matlab')
        self.mat_file_name = configuration_params['file_name']
        self.mat_data_name = configuration_params['data_name']
        self.columns_name = configuration_params['columns_name']

    def transform_matlab_to_dataframe(self) -> pd.DataFrame:
        """
        Organize and store data from matlab file as a dataframe.
        :return: data as a dataframe
        """
        matlab_data = self.load_matlab_file(folder_path=self.matlab_folder_path,
                                            file_name=self.mat_file_name)
        try:
            signals_data = matlab_data.get(self.mat_data_name).transpose()

            df_signals = pd.DataFrame(data=signals_data,
                                      columns=self.columns_name)

            return df_signals

        except Exception as e:
            print(f'Error converting matlab data to dataframe {e.args[1]}')

    @staticmethod
    def load_matlab_file(folder_path: str, file_name: str) -> dict:
        """
        Read matlab file with array structure.
        :param folder_path: folder where the matlab file is located
        :param file_name: file name to read
        :return: data file
        """
        try:
            # Matlab folder and file should always exist
            matlab_file_path = os.path.join(folder_path, file_name)
            matlab_data = hdf5storage.loadmat(file_name=matlab_file_path)
            return matlab_data

        except Exception as e:
            print(f'Error loading matlab file {e.args[1]}')
