import os
from abc import ABC, abstractmethod

import pandas as pd

from source.database_controller import DatabaseController
from source.furuta_utils import read_yaml_parameters, save_file_as_parquet
from source.matlab_files_controller import MatlabFilesController


class DataInterface(ABC):
    def __init__(self):
        self.matlab_files_controller = MatlabFilesController()
        self.database_controller = DatabaseController(create_db_connection=True)

    @abstractmethod
    def read_matlab_file(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        pass

    @abstractmethod
    def read_data_from_database(self, table_name: str) -> pd.DataFrame:
        pass


class DataReader(DataInterface):
    def __init__(self, save_data_flag: bool = False):
        super().__init__()
        configuration_params = read_yaml_parameters().get('data_reader')
        save_dir = configuration_params['save_path']
        save_file_name = configuration_params['file_name']
        self.save_path = os.path.join(save_dir, save_file_name)
        self.db_table = configuration_params['table']
        self.columns_to_get = configuration_params['columns']
        self.save_data_flag = save_data_flag

    def read_matlab_file(self) -> pd.DataFrame:
        raise NotImplementedError('DataReader can only access database.')

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        raise NotImplementedError('DataReader must read data from database not insert it.')

    def read_data_from_database(self) -> pd.DataFrame:
        data = self.database_controller.read_data(table_name=self.db_table,
                                                  columns=self.columns_to_get)
        return data

    def run(self) -> pd.DataFrame:
        data = self.read_data_from_database()

        if self.save_data_flag:
            save_file_as_parquet(data=data,
                                 save_path=self.save_path)

        return data


class DataInserter(DataInterface):
    def __init__(self):
        super().__init__()

    def read_matlab_file(self) -> pd.DataFrame:
        dataframe = self.matlab_files_controller.get_signals_data()
        return dataframe

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        self.database_controller.insert_data(table_name=table_name,
                                             data=data)

    def read_data_from_database(self, table_name: str) -> pd.DataFrame:
        raise NotImplementedError('DataInserter can not read data from database, only inserte it.')

    def run(self, table_name) -> None:
        data_from_matlab = self.read_matlab_file()
        self.insert_data_to_database(data=data_from_matlab,
                                     table_name=table_name)


if __name__ == '__main__':
    data_reader = DataReader()
    data_reader.run()