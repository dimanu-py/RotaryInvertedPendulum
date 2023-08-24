from source.deep_learning.dataset_creator import DatasetCreator
from source.deep_learning.model_creator import DLModel
from source.deep_learning.model_trainer import Trainer
from source.helpers.configuration_builder import (Configuration,
                                                  DatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder,
                                                  TrainingConfigurationBuilder)
from source.workers.design_system import DesignSystem


class FurutaPendulum(DesignSystem):

    def __init__(self):
        self.model = None

        self.dataset_configuration = Configuration(builder=DatasetConfigurationBuilder())
        self.neural_network_configuration = Configuration(builder=NeuralNetworkConfigurationBuilder())
        self.training_configuration = Configuration(builder=TrainingConfigurationBuilder())

    def create_dataset(self):
        dataset_creator = DatasetCreator(configuration=self.dataset_configuration)

        dataset = dataset_creator.create_training_dataset()
        dataset_creator.save_dataset(dataset)
        return dataset

    def create_model(self):
        self.model = DLModel(configuration=self.neural_network_configuration)
        self.model.create_model_architecture()
        self.model.compile_model()
        return self.model

    def training(self, model, dataset):
        trainer = Trainer(configuration=self.training_configuration)
        trainer.train(model, dataset)

    def evaluate(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    furuta_pendulum = FurutaPendulum()
    training_data = furuta_pendulum.create_dataset()
    neural_network = furuta_pendulum.create_model()
    furuta_pendulum.training(neural_network, training_data)
