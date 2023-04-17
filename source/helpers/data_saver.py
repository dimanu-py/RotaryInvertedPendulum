import os

import pandas as pd


class SaveParquet:
    """
    Class to save data into a parquet file.
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def save_file(self, dataframe: pd.DataFrame, save_file_name: str) -> None:
        """
        Save a dataframe with parquet extension.
        :param dataframe: dataframe to save
        :param save_file_name: name of the file to be saved
        """
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        full_path = f'{self.folder_path}\\{save_file_name}.parquet'

        try:
            dataframe.to_parquet(path=full_path,
                                 index=False)

        except Exception as e:
            print(f'Error saving data as parquet {e.args[1]}')


class SavePickle:
    """
    Class to save data into a pickle file.
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def save_file(self, dataframe: pd.DataFrame, save_file_name: str) -> None:
        """
        Save a dataframe with pickle extension.
        :param dataframe: dataframe to save
        :param save_file_name: name of the file to be saved
        """
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        full_path = f'{self.folder_path}\\{save_file_name}.pickle'

        try:
            dataframe.to_pickle(path=full_path,
                                index=False)

        except Exception as e:
            print(f'Error saving data as parquet {e.args[1]}')
