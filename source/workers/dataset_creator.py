import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.data_saver import SaveFile


class DatasetCreator:
    """
    Gather data from matlab files and save them into a file inside data/datasets folder
    """
    def __init__(self, matlab_controller: MatlabFilesController, data_saver: SaveFile) -> None:
        self.matlab_controller = matlab_controller
        self.file_saver = data_saver

    def create_dataset(self, matlab_file_name: str, matlab_data_name: str, selected_columns: list, save_file_name: str) -> None:
        """
        Main method. Gets data from matlab files and save them into a file.
        """
        matlab_data = self.get_data_from_matlab(data_file_name=matlab_file_name,
                                                matlab_data_name=matlab_data_name,
                                                selected_columns=selected_columns)
        self.save_data(dataframe=matlab_data,
                       file_name=save_file_name)

    def get_data_from_matlab(self, data_file_name: str, matlab_data_name: str, selected_columns: list) -> pd.DataFrame:
        """
        Get data from matlab file.
        """
        data = self.matlab_controller.transform_matlab_to_dataframe(data_file_name=data_file_name,
                                                                    mat_data_name=matlab_data_name,
                                                                    columns_name=selected_columns)
        return data

    def save_data(self, dataframe: pd.DataFrame, file_name: str) -> None:
        """
        Save data into a file.
        """
        self.file_saver.save_file(dataframe=dataframe,
                                  save_file_name=file_name)
