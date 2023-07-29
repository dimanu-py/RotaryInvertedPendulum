from source.env_settings import EnvSettings
from source.helpers.data_saver import SaveParquet
from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.configuration_builder import Configuration, RawDatasetConfigurationBuilder
from source.workers.dataset_creator import DatasetCreator

env_vars = EnvSettings.get_env_vars()

MATLAB_PATH = env_vars.MATLAB_PATH
DATASETS_PATH = env_vars.DATASETS_PATH
PARAMETERS_PATH = env_vars.PARAMS_PATH


class FurutaPendulum:

    def __init__(self):
        self.matlab_controller = MatlabFilesController(matlab_folder=MATLAB_PATH)
        self.dataset_saver = SaveParquet(folder_path=DATASETS_PATH)

    def create_dataset(self):
        raw_dataset_configuration = Configuration(builder=RawDatasetConfigurationBuilder())
        dataset_creator = DatasetCreator(matlab_controller=self.matlab_controller,
                                         data_saver=self.dataset_saver,
                                         configuration=raw_dataset_configuration)

        dataset_creator.create_dataset(parameters_path=PARAMETERS_PATH)

    def build_model(self):
        model_creator = ModelFactory()  # TODO: poner tipo de red que quiero crear?

        model_creator.build_model(configuration=None)  # TODO: cambiar None por la configuration correspondiente

    def build_model(self):
        model_creator = ModelFactory()  # TODO: poner tipo de red que quiero crear?

        model_creator.build_model(configuration=None)  # TODO: cambiar None por la configuration correspondiente

if __name__ == '__main__':
    furuta_pendulum = FurutaPendulum()
    furuta_pendulum.create_dataset()
