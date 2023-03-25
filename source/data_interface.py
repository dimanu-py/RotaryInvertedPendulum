import pandas as pd

from source.database_controller import DatabaseController
from source.matlab_files_controller import MatlabFilesController


class DataInterface:
    def __init__(self):
        self.matlab_files_controller = MatlabFilesController()
        self.database_controller = DatabaseController(create_db_connection=True)

    def read_matlab_file(self) -> pd.DataFrame:
        dataframe = self.matlab_files_controller.get_signals_data()
        return dataframe

    def insert_data_to_database(self, data: pd.DataFrame, table_name: str) -> None:
        self.database_controller.insert_data(table_name=table_name,
                                             data=data)

    def read_data_from_database(self, table_name: str) -> pd.DataFrame:
        data = self.database_controller.read_data(table_name=table_name)
        return data
