import os

import pandas as pd


class SaveFile:
    """
    Class to save data into a file.
    The class decides which file format to use based on the extension.
    """
    def __init__(self, folder_path: str) -> None:
        self.save_folder_path = folder_path

        self.save_funcs = {'parquet': SaveParquet,
                           'pickle': SavePickle}

    def save_file(self, dataframe: pd.DataFrame, extension: str, file_name: str) -> None:
        """
        Save a dataframe with a specified file format.
        :param dataframe: dataframe to save
        :param extension: file format to save
        :param file_name: name of the file to be saved
        """
        selected_saver = self.save_funcs.get(extension)

        if not selected_saver:
            raise ValueError(f"Invalid file format {extension}")

        saver_instance = selected_saver(self.save_folder_path)
        saver_instance.save_file(dataframe=dataframe,
                                 save_file_name=file_name)


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
