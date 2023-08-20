from source.deep_learning.dataset_creator import DatasetCreator
from source.deep_learning.dl_model_creator import FullyConnectedNetwork
from source.helpers.configuration_builder import (Configuration,
                                                  RawDatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder)
from source.helpers.data_saver import SaveParquet
from source.helpers.matlab_data_converter import MatlabDataConverter


class FurutaPendulum:

    def __init__(self):
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
        model = FullyConnectedNetwork(configuration=self.neural_network_configuration)
        model.create_architecture()


if __name__ == '__main__':
    import time
    start_time = time.time()
    furuta_pendulum = FurutaPendulum()
    stop_time = time.time() - start_time
    print(f"Time elapsed: {stop_time}")
    training_data = furuta_pendulum.create_dataset()
    furuta_pendulum.create_model()
