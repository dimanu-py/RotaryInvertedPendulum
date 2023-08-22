import hdf5storage
import pandas as pd

from typing import List, Dict


class MatlabDataConverter:
    """
    Class to read data from a matlab file.
    """
    MATLAB_FOLDER = "../data/matlab"

    def transform_matlab_to_dataframe(self, data_file_name: str, mat_data_name: str, data_source: str, columns_name: List[str]) -> pd.DataFrame:
        """
        Stores data from matlab file as a dataframe.
        """
        matlab_data = self._load_matlab_file(mat_file_name=data_file_name,
                                             mat_data_source=data_source)
        try:
            signals_data = matlab_data.get(mat_data_name).transpose()

            df_signals = pd.DataFrame(data=signals_data,
                                      columns=columns_name)

            return df_signals

        except Exception as error:
            print(f'Error converting matlab data to dataframe -> {error.args[1]}')

    def _load_matlab_file(self, mat_file_name: str, mat_data_source: str) -> Dict:
        """
        Read matlab file with array structure.
        """
        try:
            # TODO: find better way to create paths
            matlab_file_path = f'{self.MATLAB_FOLDER}/{mat_data_source}/{mat_file_name}'
            matlab_data = hdf5storage.loadmat(file_name=matlab_file_path)
            return matlab_data

        except Exception as e:
            print(f'Error loading matlab file -> {e.args[1]}')
