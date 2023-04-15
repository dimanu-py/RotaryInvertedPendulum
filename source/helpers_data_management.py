import pandas as pd

from source.env_settings import EnvSettings
from helpers.matlab_files_controller import MatlabFilesController
from helpers.data_saver import SaveFile


env_vars = EnvSettings.get_settings()


class DataSaverHelperRun:

    def __init__(self, matlab_controller: MatlabFilesController, file_saver: SaveFile):
        self.matlab_controller = matlab_controller
        self.file_saver = file_saver

    def run(self) -> None:
        """
        Run the helper.
        :return:
        """
        matlab_data = self.get_data_from_matlab()
        self.save_data(dataframe=matlab_data)

    def get_data_from_matlab(self) -> pd.DataFrame:
        """
        Get data from matlab file.
        :return: matlab data store as a dataframe
        """
        data = self.matlab_controller.transform_matlab_to_dataframe()
        return data

    def save_data(self, dataframe: pd.DataFrame) -> None:
        """
        Save data into a file.
        :param dataframe: data to save
        :return:
        """
        self.file_saver.save_file(dataframe=dataframe)


if __name__ == '__main__':

    matlab = MatlabFilesController(matlab_folder=env_vars.MATLAB_PATH)
    dataset_saver = SaveFile(folder_path=env_vars.DATASETS_PATH)

    helper_runner = DataSaverHelperRun(matlab_controller=matlab,
                                       file_saver=dataset_saver)
    helper_runner.run()
