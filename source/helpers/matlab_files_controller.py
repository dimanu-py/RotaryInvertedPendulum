import os

import hdf5storage
import pandas as pd


class MatlabFilesController:
    """
    Class to read data from a matlab file.
    The location of the file is outside the project folder.
    """
    def __init__(self, matlab_folder: str, config_yaml_path: str) -> None:
        self.matlab_folder_path = matlab_folder
        self.config_yaml_path = config_yaml_path

    def transform_matlab_to_dataframe(self, data_file_name: str, mat_data_name: str, columns_name: list[str]) -> pd.DataFrame:
        """
        Stores data from matlab file as a dataframe.
        """
        matlab_data = self._load_matlab_file(mat_file_name=data_file_name)
        try:
            signals_data = matlab_data.get(mat_data_name).transpose()

            df_signals = pd.DataFrame(data=signals_data,
                                      columns=columns_name)

            return df_signals

        except Exception as error:
            print(f'Error converting matlab data to dataframe -> {error.args[1]}')

    def _load_matlab_file(self, mat_file_name: str) -> dict:
        """
        Read matlab file with array structure.
        """
        try:
            # Matlab folder and file should always exist
            matlab_file_path = os.path.join(self.matlab_folder_path,
                                            mat_file_name)
            matlab_data = hdf5storage.loadmat(file_name=matlab_file_path)
            return matlab_data

        except Exception as e:
            print(f'Error loading matlab file -> {e.args[1]}')
