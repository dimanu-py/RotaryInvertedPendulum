import pandas as pd

from source.helpers.data_loader import LoadPickle, LoadParquet


class LoaderFactory:
    """
    Class to load raw data from a file.
    The class decides which loader to use based on the file extension.
    """
    loader_funcs = {'parquet': LoadParquet,
                    'pickle': LoadPickle}

    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def load_data(self, file_name: str, extension: str) -> pd.DataFrame:
        """
        Load a file into a dataframe.
        :param file_name: name of the file to load
        :param extension: file format to load
        """
        selected_loader = LoaderFactory.loader_funcs.get(extension)

        if not selected_loader:
            raise ValueError(f"Invalid file format {extension}")

        loader_instance = selected_loader(self.folder_path)
        data_loaded = loader_instance.load_data(file_name=file_name)

        return data_loaded
