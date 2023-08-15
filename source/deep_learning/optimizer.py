import tensorflow as tf
from abc import abstractmethod, ABC

from source.helpers.configuration_builder import Configuration


class Optimizer:
    """Class to encapsulate the optimizer"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.optimizer = self.create_optimizer()

    def create_optimizer(self) -> "Optimizer":
        optimizer_type = self.configuration.optimizer_config.optimizer_type
        optimizer_classes = {'adam': AdamOptimizer,
                             'sgd': SGDOptimizer,
                             'rmsprop': RMSPropOptimizer}

        try:
            optimizer = optimizer_classes[optimizer_type](configuration=self.configuration)
            return optimizer
        except KeyError:
            print(f'Optimizer {optimizer_type} not implemented yet.')

    def get_optimizer(self) -> "Optimizer":
        return self.optimizer


# TODO: try to create some logic to be able to create the optimizers with a variable number of parameters
class OptimizerFactory(ABC):
    """Abstract class to encapsulate the configuration of the optimizer"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.optimizer = self.create_optimizer()

    @abstractmethod
    def create_optimizer(self):
        pass


class AdamOptimizer(OptimizerFactory):
    """Class to encapsulate the configuration of the Adam optimizer"""
    def create_optimizer(self):
        learning_rate = self.configuration.optimizer_config.learning_rate
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class SGDOptimizer(OptimizerFactory):
    """Class to encapsulate the configuration of the SGD optimizer"""
    def create_optimizer(self):
        learning_rate = self.configuration.optimizer_config.learning_rate
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


class RMSPropOptimizer(OptimizerFactory):
    """Class to encapsulate the configuration of the RMSProp optimizer"""
    def create_optimizer(self):
        learning_rate = self.configuration.optimizer_config.learning_rate
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
