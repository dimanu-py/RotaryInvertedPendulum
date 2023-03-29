import pandas as pd

from source.data_interface import DataInterface


class DataInserter(DataInterface):
    def __init__(self):
        super().__init__()

    def read_matlab_file(self) -> pd.DataFrame:
        dataframe = self.matlab_files_controller.get_signals_data()
        return dataframe

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        self.database_controller.insert_data(table_name=table_name,
                                             data=data)

    def read_data_from_database(self) -> pd.DataFrame:
        raise NotImplementedError('DataInserter can not read data from database, only inserte it.')

    def run(self, table_name) -> None:
        data_from_matlab = self.read_matlab_file()
        self.insert_data_to_database(data=data_from_matlab,
                                     table_name=table_name)


if __name__ == '__main__':
    data_reader = DataInserter()
    data_reader.run()
