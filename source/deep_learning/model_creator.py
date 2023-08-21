from abc import ABC, abstractmethod

from source.helpers.configuration_builder import Configuration
from source.workers.architecture import FullyConnectedNetwork
from source.workers.loss_function import LossFunctionFactory
from source.workers.metrics import MetricsFactory
from source.workers.optimizer import OptimizerFactory


class AIModel(ABC):
    @abstractmethod
    def create_model_architecture(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass


class DLModel(AIModel):
    CONFIG_KEY = 'model'

    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)
        self.model = None

    def create_model_architecture(self):
        architect = FullyConnectedNetwork(configuration=self.configuration.architecture_config)
        self.model = architect.create_architecture()

    def compile_model(self):
        optimizer = OptimizerFactory.get_optimizer(configuration=self.configuration.optimizer_config)
        loss_function = LossFunctionFactory.get_loss_function(configuration=self.configuration.loss_config)
        metrics = MetricsFactory.get_metrics(configuration=self.configuration.metrics_config)

        self.model.compile(optimizer=optimizer,
                           loss_function=loss_function,
                           metrics=metrics)
