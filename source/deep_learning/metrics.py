import tensorflow as tf
from abc import abstractmethod, ABC
from typing import List

from source.helpers.configuration_builder import Configuration


class Metrics:
    """Class to encapsulate the metrics"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.metrics = self.create_metrics()

    def create_metrics(self) -> List["Metrics"]:
        """Method to create the metrics dynamically based on the configuration"""
        metrics_type = self.configuration.metrics_config.metrics
        metrics_classes = {'mae': MeanAbsoluteError,
                           'mse': MeanSquaredError,
                           'rmse': RootMeanSquaredError,
                           'mape': MeanAbsolutePercentageError,
                           'msle': MeanSquaredLogarithmicError,
                           'r2': R2Score}

        try:
            metrics = [metrics_classes[metric](self.configuration) for metric in metrics_type]
            return metrics
        except KeyError:
            print(f'Metrics {metrics_type} not implemented yet.')

    def get_metrics(self):
        """Method to get the metrics without accessing the metrics attribute directly"""
        return self.metrics


class MetricsFactory(ABC):
    """Abstract class to encapsulate the configuration of the metrics"""
    def __init__(self, configuration: "Configuration") -> None:
        self.configuration = configuration
        self.metrics = self.create_metrics()

    @abstractmethod
    def create_metrics(self):
        pass


class MeanAbsoluteError(MetricsFactory):
    """Class to encapsulate the configuration of the Mean Absolute Error metric"""
    def create_metrics(self):
        return tf.keras.metrics.MeanAbsoluteError()


class MeanSquaredError(MetricsFactory):
    """Class to encapsulate the configuration of the Mean Squared Error metric"""
    def create_metrics(self):
        return tf.keras.metrics.MeanSquaredError()


class RootMeanSquaredError(MetricsFactory):
    """Class to encapsulate the configuration of the Root Mean Squared Error metric"""
    def create_metrics(self):
        return tf.keras.metrics.RootMeanSquaredError()


class MeanAbsolutePercentageError(MetricsFactory):
    """Class to encapsulate the configuration of the Mean Absolute Percentage Error metric"""
    def create_metrics(self):
        return tf.keras.metrics.MeanAbsolutePercentageError()


class MeanSquaredLogarithmicError(MetricsFactory):
    """Class to encapsulate the configuration of the Mean Squared Logarithmic Error metric"""
    def create_metrics(self):
        return tf.keras.metrics.MeanSquaredLogarithmicError()


class R2Score(MetricsFactory):
    """Class to encapsulate the configuration of the R2 Score metric"""
    def create_metrics(self):
        return tf.keras.metrics.R2Score()
