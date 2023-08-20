import abc
import os

import pandas as pd


class SaveFile(abc.ABC):
    """
    Abstract class to save data into a file.
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @abc.abstractmethod
    def save_file(self, dataframe: pd.DataFrame, save_file_name: str) -> None:
        """
        Abstract method to save a dataframe.
        """
        pass

    def check_folder_exists(self) -> None:
        """
        Check if the folder exists. If not, create it.
        """
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)


class SaveParquet(SaveFile):
    """
    Concrete class to save data into a parquet file.
    """
    def save_file(self, dataframe: pd.DataFrame, save_file_name: str) -> None:
        """
        Save a dataframe with parquet extension.
        """
        full_path = self.build_save_path(save_file_name)

        try:
            dataframe.to_parquet(path=full_path,
                                 index=False)

        except Exception as e:
            print(f'Error saving data as parquet {e.args[1]}')

    def build_save_path(self, save_file_name: str) -> str:
        """
        Build the full path to save the file.
        """
        self.check_folder_exists()
        full_path = f'{self.folder_path}/{save_file_name}.parquet'
        return full_path
