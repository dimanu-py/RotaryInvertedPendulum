import pandas as pd

from source.helpers.furuta_utils import read_yaml_parameters


class LoadData:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

        configuration_params = read_yaml_parameters().get('data_reader')
        self.file_name = configuration_params['file_name']
        self.extension = configuration_params['extension'].lower()

        self.load_funcs = {'parquet': LoadParquet,
                           'pickle': LoadPickle}

    def load_file(self) -> pd.DataFrame:
        """
        Load a file into a dataframe.
        """
        selected_loader = self.load_funcs.get(self.extension)

        if not selected_loader:
            raise ValueError(f"Invalid file format {self.extension}")

        loader_instance = selected_loader(self.folder_path)
        data_loaded = loader_instance.load_file(file_name=self.file_name)

        return data_loaded


class LoadParquet:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def load_file(self, file_name: str) -> pd.DataFrame:
        """
        Load a file with parquet extension into a dataframe.
        :param file_name: name of the file to load
        """
        full_path = f'{self.folder_path}\\{file_name}.parquet'

        try:
            data = pd.read_parquet(path=full_path)
            return data
        except Exception as e:
            print(f'Error loading parquet data {e.args[1]}')


class LoadPickle:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def load_file(self, file_name: str) -> pd.DataFrame:
        """
        Load a file with pickle extension into a dataframe.
        :param file_name: name of the file to load
        """
        full_path = f'{self.folder_path}\\{file_name}.pickle'

        try:
            data = pd.read_pickle(filepath_or_buffer=full_path)
            return data

        except Exception as e:
            print(f'Error loading pickle data {e.args[1]}')
