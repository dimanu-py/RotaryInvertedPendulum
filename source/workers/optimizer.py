from abc import abstractmethod, ABC

from keras.optimizers import (Adam,
                              SGD,
                              RMSprop)

from source.helpers.configuration_builder import Configuration


class OptimizerFactory:
    """Class to encapsulate the optimizer"""
    @staticmethod
    def get_optimizer(configuration: "Configuration") -> "Optimizer":
        """Method to create the optimizer dynamically based on the configuration"""
        optimizer_type = configuration.optimizer_type
        optimizer_classes = {'adam': AdamOptimizer,
                             'sgd': SGDOptimizer,
                             'rmsprop': RMSPropOptimizer}

        try:
            optimizer = optimizer_classes[optimizer_type](configuration=configuration)
            return optimizer.optimizer
        except KeyError:
            print(f'Optimizer {optimizer_type} not implemented yet.')


# TODO: try to create some logic to be able to create the optimizers with a variable number of parameters
class Optimizer(ABC):
    """Abstract class to encapsulate the configuration of the optimizer"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.optimizer = self.configure_optimizer()

    @abstractmethod
    def configure_optimizer(self):
        pass


class AdamOptimizer(Optimizer):
    """Class to encapsulate the configuration of the Adam optimizer"""
    def configure_optimizer(self):
        learning_rate = self.configuration.learning_rate
        return Adam(learning_rate=learning_rate)


class SGDOptimizer(Optimizer):
    """Class to encapsulate the configuration of the SGD optimizer"""
    def configure_optimizer(self):
        learning_rate = self.configuration.learning_rate
        return SGD(learning_rate=learning_rate)


class RMSPropOptimizer(Optimizer):
    """Class to encapsulate the configuration of the RMSProp optimizer"""
    def configure_optimizer(self):
        learning_rate = self.configuration.learning_rate
        return RMSprop(learning_rate=learning_rate)
