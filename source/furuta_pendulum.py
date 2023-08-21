from source.deep_learning.dataset_creator import DatasetCreator
from source.helpers.configuration_builder import (Configuration,
                                                  RawDatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder)
from source.helpers.data_saver import SaveParquet
from source.helpers.matlab_data_converter import MatlabDataConverter
from source.workers.design_system import DesignSystem


class FurutaPendulum(DesignSystem):

    def __init__(self):
        self.model = None

        self.matlab_converter = MatlabDataConverter()
        self.dataset_saver = SaveParquet()
        self.raw_dataset_configuration = Configuration(builder=RawDatasetConfigurationBuilder())
        self.neural_network_configuration = Configuration(builder=NeuralNetworkConfigurationBuilder())

    def create_dataset(self):
        dataset_creator = DatasetCreator(matlab_converter=self.matlab_converter,
                                         configuration=self.raw_dataset_configuration)

        dataset = dataset_creator.create_dataset()
        return dataset

    def create_model(self):
        pass

    def training(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    furuta_pendulum = FurutaPendulum()
    training_data = furuta_pendulum.create_dataset()
    furuta_pendulum.create_model()
