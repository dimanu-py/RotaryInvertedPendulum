import pandas as pd


class LoadData:
    """
    Class to load raw data from a file.
    The class decides which loader to use based on the file extension.
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

        self.load_funcs = {'parquet': LoadParquet,
                           'pickle': LoadPickle}

    def load_file(self, file_name: str, extension: str) -> pd.DataFrame:
        """
        Load a file into a dataframe.
        :param file_name: name of the file to load
        :param extension: file format to load
        """
        selected_loader = self.load_funcs.get(extension)

        if not selected_loader:
            raise ValueError(f"Invalid file format {extension}")

        loader_instance = selected_loader(self.folder_path)
        data_loaded = loader_instance.load_file(file_name=file_name)

        return data_loaded


class LoadParquet:
    """
    Class to load parquet files into a dataframe.
    """
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
    """"
    Class to load pickle files into a dataframe.
    """
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
