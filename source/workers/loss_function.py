from keras.losses import (MeanSquaredError,
                          MeanAbsoluteError,
                          MeanSquaredLogarithmicError)
from abc import abstractmethod, ABC

from source.helpers.configuration_builder import Configuration


class LossFunctionFactory:
    """Class to encapsulate the loss function"""
    @staticmethod
    def get_loss_function(configuration) -> "LossFunction":
        """Method to create the loss function dynamically based on the configuration"""
        loss_function_type, = configuration.loss
        loss_function_classes = {'mse': MSE,
                                 'mae': MAE,
                                 'msle': MSLE}

        try:
            loss_function = loss_function_classes[loss_function_type](configuration=configuration)
            return loss_function.loss_function
        except KeyError:
            print(f'Loss function {loss_function_type} not implemented yet.')


# TODO: try to create some logic to be able to create the loss function with a variable number of parameters
class LossFunction(ABC):
    """Abstract class to encapsulate the configuration of the loss function"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.loss_function = self.configure_loss_function()

    @abstractmethod
    def configure_loss_function(self):
        pass


class MSE(LossFunction):
    """Class to encapsulate the configuration of the mean squared error loss function"""
    def configure_loss_function(self):
        return MeanSquaredError()


class MAE(LossFunction):
    """Class to encapsulate the configuration of the mean absolute error loss function"""
    def configure_loss_function(self):
        return MeanAbsoluteError()


class MSLE(LossFunction):
    """Class to encapsulate the configuration of the mean squared logarithmic error loss function"""
    def configure_loss_function(self):
        return MeanSquaredLogarithmicError()
