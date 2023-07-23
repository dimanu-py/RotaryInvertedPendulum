import pandas as pd
from abc import abstractmethod, ABC


class DataLoader(metaclass=ABC):

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @abstractmethod
    def load_data(self, file_name: str) -> pd.DataFrame:
        pass


class LoadParquet(DataLoader):
    """
    Class to load parquet files into a dataframe.
    """
    def __init__(self, folder_path: str) -> None:
        super().__init__(folder_path=folder_path)

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load a file with parquet extension into a dataframe.
        """
        full_path = f'{self.folder_path}\\{file_name}'

        try:
            data = pd.read_parquet(path=full_path)
            return data
        except Exception as e:
            print(f'Error loading parquet data {e.args[1]}')


class LoadPickle:
    """"
    Class to load pickle files into a dataframe.
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load a file with pickle extension into a dataframe.
        :param file_name: name of the file to load
        """
        full_path = f'{self.folder_path}\\{file_name}'

        try:
            data = pd.read_pickle(filepath_or_buffer=full_path)
            return data

        except Exception as e:
            print(f'Error loading pickle data {e.args[1]}')
