import pandas as pd

from source.data_interface import DataInterface
from source.furuta_utils import read_yaml_parameters, save_file_as_parquet


class DataReader(DataInterface):
    def __init__(self, save_data_flag: bool = False):
        super().__init__()
        configuration_params = read_yaml_parameters().get('data_reader')
        self.save_dir = configuration_params['save_path']
        self.save_file_name = configuration_params['file_name']
        self.db_table = configuration_params['table']
        self.columns_to_get = configuration_params['columns']
        self.save_data_flag = save_data_flag

    def read_matlab_file(self) -> pd.DataFrame:
        raise NotImplementedError('DataReader can only access database.')

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        raise NotImplementedError('DataReader must read data from database not insert it.')

    def read_data_from_database(self, table_name: str, columns: list[str]) -> pd.DataFrame:
        data = self.database_controller.read_data(table_name=table_name,
                                                  columns=columns)
        return data

    def run(self) -> pd.DataFrame:
        data = self.read_data_from_database(table_name=self.db_table,
                                            columns=self.columns_to_get)

        if self.save_data_flag:
            save_file_as_parquet(data=data,
                                 save_dir=self.save_dir,
                                 save_file_name=self.save_file_name)

        return data


if __name__ == '__main__':
    data_reader = DataReader()
    data_reader.run()
