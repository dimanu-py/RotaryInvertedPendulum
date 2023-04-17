import pandas as pd

from source.helpers.data_saver import SavePickle, SaveParquet


class SaverFactory:
    """
    Class to save data into a file.
    The class decides which file format to use based on the extension.
    """
    saver_funcs = {'parquet': SaveParquet,
                   'pickle': SavePickle}

    def __init__(self, folder_path: str) -> None:
        self.save_folder_path = folder_path

    def save_file(self, dataframe: pd.DataFrame, extension: str, file_name: str) -> None:
        """
        Save a dataframe with a specified file format.
        :param dataframe: dataframe to save
        :param extension: file format to save
        :param file_name: name of the file to be saved
        """
        selected_saver = SaverFactory.saver_funcs.get(extension)

        if not selected_saver:
            raise ValueError(f"Invalid file format {extension}")

        saver_instance = selected_saver(self.save_folder_path)
        saver_instance.save_file(dataframe=dataframe,
                                 save_file_name=file_name)
