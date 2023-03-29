from abc import ABC, abstractmethod

import pandas as pd

from source.database_controller import DatabaseController
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
    def read_data_from_database(self) -> pd.DataFrame:
        pass
