from source.deep_learning.dataset_creator import DatasetCreator
from source.deep_learning.model_creator import DLModel
from source.helpers.configuration_builder import (Configuration,
                                                  RawDatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder)
from source.workers.design_system import DesignSystem


class FurutaPendulum(DesignSystem):

    def __init__(self):
        self.model = None

        self.raw_dataset_configuration = Configuration(builder=RawDatasetConfigurationBuilder())
        self.neural_network_configuration = Configuration(builder=NeuralNetworkConfigurationBuilder())

    def create_dataset(self):
        dataset_creator = DatasetCreator(configuration=self.raw_dataset_configuration)

        dataset = dataset_creator.create_dataset()
        dataset_creator.save_dataset(dataset)
        return dataset

    def create_model(self):
        self.model = DLModel(configuration=self.neural_network_configuration)
        self.model.create_model_architecture()
        self.model.compile_model()

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
