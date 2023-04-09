import pandas as pd

from source.env_settings import EnvSettings
from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.data_saver import SaveFile


# TODO: acceder a las variables de entorno para los paths
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


if __name__ == '__main__':
    env_vars = EnvSettings.get_settings()

    matlab = MatlabFilesController(matlab_folder=env_vars.MATLAB_PATH)
    dataset_saver = SaveFile(folder_path=env_vars.DATASETS_PATH)

    helper_runner = HelpersRun(matlab_controller=matlab,
                               file_saver=dataset_saver)
    helper_runner.run()
