import tensorflow as tf
from abc import abstractmethod, ABC

from source.helpers.configuration_builder import Configuration


class LossFunction:
    """Class to encapsulate the loss function"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.loss_function = self.create_loss_function()

    def create_loss_function(self) -> "LossFunction":
        """Method to create the loss function dynamically based on the configuration"""
        loss_function_type = self.configuration.loss_function_config.loss
        loss_function_classes = {'mse': MeanSquaredError,
                                 'mae': MeanAbsoluteError,
                                 'msle': MeanSquaredLogarithmicError}

        try:
            loss_function = loss_function_classes[loss_function_type](configuration=self.configuration)
            return loss_function
        except KeyError:
            print(f'Loss function {loss_function_type} not implemented yet.')

    def get_loss_function(self) -> "LossFunction":
        """Method to get the loss function without accessing the loss_function attribute directly"""
        return self.loss_function


class LossFunctionFactory(ABC):
    """Abstract class to encapsulate the configuration of the loss function"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.loss_function = self.create_loss_function()

    @abstractmethod
    def create_loss_function(self):
        pass


class MeanSquaredError(LossFunctionFactory):
    """Class to encapsulate the configuration of the mean squared error loss function"""
    def create_loss_function(self):
        return tf.keras.losses.MeanSquaredError()


class MeanAbsoluteError(LossFunctionFactory):
    """Class to encapsulate the configuration of the mean absolute error loss function"""
    def create_loss_function(self):
        return tf.keras.losses.MeanAbsoluteError()


class MeanSquaredLogarithmicError(LossFunctionFactory):
    """Class to encapsulate the configuration of the mean squared logarithmic error loss function"""
    def create_loss_function(self):
        return tf.keras.losses.MeanSquaredLogarithmicError()
