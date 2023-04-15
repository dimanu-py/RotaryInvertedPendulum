import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.data_saver import SaveFile
from source.helpers.furuta_utils import read_yaml_parameters


class DataInserter:
    """
    Gather data from matlab files and save them into a file inside data/datasets folder
    """
    def __init__(self, matlab_controller: MatlabFilesController, file_saver: SaveFile):
        self.matlab_controller = matlab_controller
        self.file_saver = file_saver

        configuration_params = read_yaml_parameters().get('inserter')
        self.file_name = configuration_params['file_name']
        self.extension = configuration_params['extension']

    def create_dataset(self) -> None:
        """
        Main method. Gets data from matlab files and save them into a file.
        :return:
        """
        matlab_data = self.get_data_from_matlab()
        self.save_data(dataframe=matlab_data,
                       file_name=self.file_name,
                       extension=self.extension)

    def get_data_from_matlab(self) -> pd.DataFrame:
        """
        Get data from matlab file.
        :return: matlab data store as a dataframe
        """
        data = self.matlab_controller._transform_matlab_to_dataframe()
        return data

    def save_data(self, dataframe: pd.DataFrame, file_name:str, extension: str) -> None:
        """
        Save data into a file.
        :param dataframe: data to save
        :param file_name: name of the file
        :param extension: extension of the file
        :return:
        """
        self.file_saver.save_file(dataframe=dataframe,
                                  file_name=file_name,
                                  extension=extension)
