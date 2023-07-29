from source.env_settings import EnvSettings
from source.helpers.data_saver import SaveParquet
from source.helpers.matlab_files_controller import MatlabFilesController
from source.workers.dataset_creator import DatasetCreator
from source.helpers.configuration_reader import RawDatasetConfiguration

env_vars = EnvSettings.get_env_vars()


class FurutaPendulum:

    MATLAB_PATH = env_vars.MATLAB_PATH
    DATASETS_PATH = env_vars.DATASETS_PATH
    PARAMETERS_PATH = env_vars.PARAMS_PATH

    def __init__(self):
        self.matlab_controller = MatlabFilesController(matlab_folder=self.MATLAB_PATH)
        self.dataset_saver = SaveParquet(folder_path=self.DATASETS_PATH)
        self.configuration = RawDatasetConfiguration(configuration_path=self.PARAMETERS_PATH)

    def create_dataset(self):
        dataset_creator = DatasetCreator(matlab_controller=self.matlab_controller,
                                         data_saver=self.dataset_saver)

        dataset_creator.create_dataset(configuration=self.configuration)

    def build_model(self):
        model_creator = ModelFactory()  # TODO: poner tipo de red que quiero crear?

        model_creator.build_model(configuration=None)  # TODO: cambiar None por la configuration correspondiente

if __name__ == '__main__':
    furuta_pendulum = FurutaPendulum()
    furuta_pendulum.create_dataset()
