import abc
import os

import pandas as pd


#TODO: check strategy to save files, specially to create the path
class SaveFile(abc.ABC):
    """
    Abstract class to save data into a file.
    """
    SAVE_FOLDER = "../data"

    @abc.abstractmethod
    def save_file(self, dataframe: pd.DataFrame, folder_to_save: str, save_file_name: str) -> None:
        """
        Abstract method to save a dataframe.
        """
        pass

    @staticmethod
    def check_folder_exists(folder_path: str) -> None:
        """
        Check if the folder exists. If not, create it.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class SaveParquet(SaveFile):
    """
    Concrete class to save data into a parquet file.
    """
    def save_file(self, dataframe: pd.DataFrame, folder_to_save: str, save_file_name: str) -> None:
        """
        Save a dataframe with parquet extension.
        """
        full_path = self.build_save_path(folder_to_save,
                                         save_file_name)

        try:
            dataframe.to_parquet(path=full_path,
                                 index=False)

        except Exception as e:
            print(f'Error saving data as parquet {e.args[1]}')

    def build_save_path(self, folder_to_save: str, save_file_name: str) -> str:
        """
        Build the full path to save the file.
        """
        self.check_folder_exists(f'{self.SAVE_FOLDER}/{folder_to_save}')
        full_path = f'{self.SAVE_FOLDER}/{folder_to_save}/{save_file_name}.parquet'
        return full_path
