import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.data_saver import SaveFile


class HelpersRun:

    def __init__(self, matlab_controller: MatlabFilesController, file_saver: SaveFile):
        self.matlab_controller = matlab_controller
        self.file_saver = file_saver

    def run(self) -> None:
        matlab_data = self.get_data_from_matlab()
        self.save_data(dataframe=matlab_data)

    def get_data_from_matlab(self) -> pd.DataFrame:
        data = self.matlab_controller.transform_matlab_to_dataframe()
        return data

    def save_data(self, dataframe: pd.DataFrame) -> None:
        self.file_saver.save_file(dataframe=dataframe)
