from abc import ABC, abstractmethod
from typing import Dict, Any
from functools import wraps

from source.furuta_utils import read_yaml_parameters


class Configuration:
    """Client class to construct a configuration object"""
    PARAMS_PATH = "../config"  # this path shouldn't change, is not configurable

    def __init__(self, builder: "ConfigurationBuilder"):
        self._builder = builder
        self.config = None

    def load_configuration_data(func):
        @wraps(func)
        def wrapper(self, data_key, *args, **kwargs):
            configuration_data = read_yaml_parameters(self.PARAMS_PATH, *args, **kwargs)
            try:
                selected_data = configuration_data[data_key]
                return func(self, selected_data)
            except KeyError as error:
                print(f'Impossible to find the key -> {error.args[0]}')

        return wrapper

    @load_configuration_data
    def construct(self, configuration_data: Dict[str, Any]) -> "Configuration":
        """"Construct a configuration object with its corresponding configuration components"""
        self.config = self._builder.build(configuration_data)
        return self

    def __getattr__(self, item):
        return getattr(self.config, item)


class ConfigurationComponent(ABC):
    """Abstract class to define a configuration component"""
    @abstractmethod
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        pass


class MatlabConfiguration(ConfigurationComponent):
    """Class to define a matlab configuration component to get data from matlab files"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.file_name = configuration_data.get('file_name')
        self.source = configuration_data.get('source')
        self.data = configuration_data.get('data')
        self.columns = configuration_data.get('columns')


class DatasetSaverConfiguration(ConfigurationComponent):
    """Class to define a dataset saver configuration component to save data into a file"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.dataset_name = configuration_data.get('dataset_name')


class ArchitectureModelConfiguration(ConfigurationComponent):
    """Class to define a build model configuration component to build a neural network model"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.type = configuration_data.get('model_type')
        self.input_shape = configuration_data.get('input_shape')
        self.number_units = configuration_data.get('number_units')
        self.activation_hidden_layers = configuration_data.get('activation_hidden_layers')
        self.activation_output_layer = configuration_data.get('activation_output_layer')


class OptimizerConfiguration(ConfigurationComponent):
    """Class to define a optimizer configuration component to optimize a neural network model"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.optimizer_type = configuration_data.get('optimizer_type')
        self.learning_rate = configuration_data.get('learning_rate')


class LossConfiguration(ConfigurationComponent):
    """Class to define a loss configuration component to define a loss function for a neural network model"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.loss = configuration_data


class MetricsConfiguration(ConfigurationComponent):
    """Class to define a metrics configuration component to define a metrics for a neural network model"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.metrics = configuration_data


class ConfigurationBuilder(ABC):
    """Abstract class to define a configuration builder"""
    @abstractmethod
    def build(self, configuration_data: Dict[str, Any]) -> None:
        pass


class RawDatasetConfigurationBuilder(ConfigurationBuilder):
    """Class to define a raw dataset configuration builder"""
    matlab_config: "ConfigurationComponent" = None
    dataset_saver_config: "ConfigurationComponent" = None

    def build(self, configuration_data: Dict[str, Any]) -> "RawDatasetConfigurationBuilder":
        """Build a raw dataset configuration object with its corresponding configuration components"""
        self.matlab_config = MatlabConfiguration(configuration_data=configuration_data['matlab'])
        self.dataset_saver_config = DatasetSaverConfiguration(configuration_data=configuration_data['saver'])
        return self


class NeuralNetworkConfigurationBuilder(ConfigurationBuilder):
    """Class to define a neural network configuration builder"""
    architecture_config: "ConfigurationComponent" = None
    optimizer_config: "ConfigurationComponent" = None
    loss_config: "ConfigurationComponent" = None
    metrics_config: "ConfigurationComponent" = None

    def build(self, configuration_data: Dict[str, Any]) -> "NeuralNetworkConfigurationBuilder":
        """Build a neural network configuration object with its corresponding configuration components"""
        self.architecture_config = ArchitectureModelConfiguration(configuration_data=configuration_data['architecture'])
        self.optimizer_config = OptimizerConfiguration(configuration_data=configuration_data['optimizer'])
        self.loss_config = LossConfiguration(configuration_data=configuration_data['loss'])
        self.metrics_config = MetricsConfiguration(configuration_data=configuration_data['metrics'])
        return self
