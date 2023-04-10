import os

import pandas as pd

from source.helpers.furuta_utils import read_yaml_parameters


class SaveFile:
    def __init__(self, folder_path: str) -> None:
        self.save_folder_path = folder_path

        configuration_params = read_yaml_parameters().get('data_saver')
        self.file_name = configuration_params['file_name']
        self.extension = configuration_params['extension'].lower()

        self.save_funcs = {'parquet': SaveParquet,
                           'pickle': SavePickle}

    def save_file(self, dataframe: pd.DataFrame) -> None:
        """
        Save a dataframe with a specified file format.
        :param dataframe: dataframe to save
        """
        selected_saver = self.save_funcs.get(self.extension)

        if not selected_saver:
            raise ValueError(f"Invalid file format {self.extension}")

        saver_instance = selected_saver(self.save_folder_path)
        saver_instance.save_file(dataframe=dataframe,
                                 save_file_name=self.file_name)


class SaveParquet:
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
