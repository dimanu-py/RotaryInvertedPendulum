import pandas as pd

from source.helpers.data_saver import SaveFile
from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.configuration_reader import RawDatasetConfiguration


class DatasetCreator:
    """
    Gather data from matlab files and save them into a file inside data/datasets folder
    """
    def __init__(self, matlab_controller: MatlabFilesController, data_saver: SaveFile) -> None:
        self.matlab_controller = matlab_controller
        self.file_saver = data_saver

    def create_dataset(self, configuration: RawDatasetConfiguration) -> None:
        """
        Main method. Gets data from matlab files and save them into a file.
        """
        matlab_file = configuration.matlab_config.file_name
        matlab_data_name = configuration.matlab_config.data
        selected_columns = configuration.matlab_config.columns

        save_file_name = configuration.dataset_saver_config.dataset_name

        matlab_data = self.get_data_from_matlab(data_file_name=matlab_file,
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
