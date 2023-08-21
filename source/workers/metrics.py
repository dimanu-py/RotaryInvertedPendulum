from keras.metrics import (MeanAbsoluteError,
                           MeanSquaredError,
                           RootMeanSquaredError,
                           R2Score,
                           MeanSquaredLogarithmicError,
                           MeanAbsolutePercentageError)
from abc import abstractmethod, ABC
from typing import List

from source.helpers.configuration_builder import Configuration


class MetricsFactory:
    """Class to encapsulate the metrics"""
    @staticmethod
    def get_metrics(configuration: "Configuration") -> List["Metrics"]:
        """Method to create the metrics dynamically based on the configuration"""
        metrics_type = configuration.metrics
        metrics_classes = {'mae': MAE,
                           'mse': MSE,
                           'rmse': RMSE,
                           'mape': MAPE,
                           'msle': MSLE,
                           'r2': R2,
                           }

        try:
            metrics = [metrics_classes[metric](configuration) for metric in metrics_type]
            return [metric.metrics for metric in metrics]
        except KeyError:
            print(f'Metrics {metrics_type} not implemented yet.')


class Metrics(ABC):
    """Abstract class to encapsulate the configuration of the metrics"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.metrics = self.configure_metrics()

    @abstractmethod
    def configure_metrics(self):
        pass


class MAE(Metrics):
    """Class to encapsulate the configuration of the Mean Absolute Error metric"""
    def configure_metrics(self):
        return MeanAbsoluteError()


class MSE(Metrics):
    """Class to encapsulate the configuration of the Mean Squared Error metric"""
    def configure_metrics(self):
        return MeanSquaredError()


class RMSE(Metrics):
    """Class to encapsulate the configuration of the Root Mean Squared Error metric"""
    def configure_metrics(self):
        return RootMeanSquaredError()


class MAPE(Metrics):
    """Class to encapsulate the configuration of the Mean Absolute Percentage Error metric"""
    def configure_metrics(self):
        return MeanAbsolutePercentageError()


class MSLE(Metrics):
    """Class to encapsulate the configuration of the Mean Squared Logarithmic Error metric"""
    def configure_metrics(self):
        return MeanSquaredLogarithmicError()


class R2(Metrics):
    """Class to encapsulate the configuration of the R2 Score metric"""
    def configure_metrics(self):
        return R2Score()
