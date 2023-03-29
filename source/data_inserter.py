import os

import pandas as pd

from source.data_interface import DataInterface
from source.furuta_utils import read_yaml_parameters


class DataInserter(DataInterface):
    def __init__(self):
        super().__init__()
        self.yaml_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config/data_inserter_config.yaml'))
        configuration_params = read_yaml_parameters(yaml_path=self.yaml_config_path).get('data_inserter')
        self.table_name = configuration_params['table']

    def read_matlab_file(self) -> pd.DataFrame:
        dataframe = self.matlab_files_controller.get_signals_data()
        return dataframe

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        self.database_controller.insert_data(table_name=table_name,
                                             data=data)

    def read_data_from_database(self, table_name: str, columns: list[str] = None) -> pd.DataFrame:
        raise NotImplementedError('DataInserter can not read data from database, only inserte it.')

    def run(self) -> None:
        data_from_matlab = self.read_matlab_file()
        self.insert_data_to_database(data=data_from_matlab,
                                     table_name=self.table_name)


if __name__ == '__main__':
    data_reader = DataInserter()
    data_reader.run()
