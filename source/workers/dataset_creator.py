import pandas as pd

from source.helpers.data_saver import SaveFile
from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.configuration_builder import Configuration


class DatasetCreator:
    """
    Gather data from matlab files and save them into a file inside data/datasets folder
    """
    def __init__(self, matlab_controller: MatlabFilesController, data_saver: SaveFile, configuration: Configuration) -> None:
        self.matlab_controller = matlab_controller
        self.file_saver = data_saver
        self.configuration = configuration

    def create_dataset(self, parameters_path: str) -> None:
        """
        Main method. Gets data from matlab files and save them into a file.
        """
        # TODO: check strategy used to create raw_dataset_configuration -> I don't want to pass parameters_path as
        #  argument from outside and don't want to hardcode raw_dataset key
        raw_dataset_configuration = self.configuration.construct(parameters_path, 'raw_dataset')
        matlab_file = raw_dataset_configuration.matlab_configuration.file_name
        matlab_data_name = raw_dataset_configuration.matlab_configuration.data
        selected_columns = raw_dataset_configuration.matlab_configuration.columns

        save_file_name = raw_dataset_configuration.dataset_saver_configuration.dataset_name

        # TODO: maybe is better to pass raw_dataset_configuration.matlab_configuration as argument instead of passing
        #  each attribute separately
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
