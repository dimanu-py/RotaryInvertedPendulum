from source.deep_learning.dataset_creator import DatasetCreator
from source.env_settings import EnvSettings
from source.helpers.configuration_builder import (Configuration,
                                                  RawDatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder)
from source.helpers.data_saver import SaveParquet
from source.helpers.matlab_files_controller import MatlabFilesController
from source.deep_learning.dl_model_creator import FullyConnectedNetwork

env_vars = EnvSettings.get_env_vars()

MATLAB_PATH = env_vars.MATLAB_PATH
DATASETS_PATH = env_vars.DATASETS_PATH
PARAMETERS_PATH = env_vars.PARAMS_PATH


class FurutaPendulum:

    def __init__(self):
        self.matlab_controller = MatlabFilesController(matlab_folder=MATLAB_PATH)
        self.dataset_saver = SaveParquet(folder_path=DATASETS_PATH)
        self.raw_dataset_configuration = Configuration(builder=RawDatasetConfigurationBuilder())
        self.neural_network_configuration = Configuration(builder=NeuralNetworkConfigurationBuilder())

    def create_dataset(self):
        dataset_creator = DatasetCreator(matlab_controller=self.matlab_controller,
                                         data_saver=self.dataset_saver,
                                         configuration=self.raw_dataset_configuration)

        dataset_creator.create_dataset(parameters_path=PARAMETERS_PATH)

    def create_model(self):
        model = FullyConnectedNetwork(configuration=self.neural_network_configuration)
        model.create_architecture()


if __name__ == '__main__':
    furuta_pendulum = FurutaPendulum()
    furuta_pendulum.create_dataset()
    furuta_pendulum.create_model()
