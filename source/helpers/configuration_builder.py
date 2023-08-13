from abc import ABC, abstractmethod
from typing import Dict, Any

from source.furuta_utils import load_configuration_data


class Configuration:
    """Client class to construct a configuration object"""
    def __init__(self, builder: "ConfigurationBuilder"):
        self._builder = builder

    @load_configuration_data
    def construct(self, configuration_data: Dict[str, Any]) -> "Configuration":
        """"Construct a configuration object with its corresponding configuration components"""
        self._builder.build(configuration_data)
        return self

    def __getattr__(self, item):
        """Method to get configuration component attribute from builder with dot notation"""
        return getattr(self._builder, item)


class ConfigurationComponent(ABC):
    """Abstract class to define a configuration component"""
    @abstractmethod
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        pass


class MatlabConfiguration(ConfigurationComponent):
    """Class to define a matlab configuration component to get data from matlab files"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.file_name = configuration_data.get('file_name')
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


class CompileModelConfiguration(ConfigurationComponent):
    """Class to define a compile model configuration component to compile a neural network model"""
    def __init__(self, configuration_data: Dict[str, Any]) -> None:
        self.loss = configuration_data.get('loss')
        self.metrics = configuration_data.get('metrics')
        self.optimizer = configuration_data.get('optimizer')
        self.learning_rate = configuration_data.get('learning_rate')


class ConfigurationBuilder(ABC):
    """Abstract class to define a configuration builder"""
    @abstractmethod
    def build(self, configuration_data: Dict[str, Any]) -> None:
        pass


class RawDatasetConfigurationBuilder(ConfigurationBuilder):
    """Class to define a raw dataset configuration builder"""
    matlab_configuration: "ConfigurationComponent" = None
    dataset_saver_configuration: "ConfigurationComponent" = None

    def build(self, configuration_data: Dict[str, Any]) -> None:
        """Build a raw dataset configuration object with its corresponding configuration components"""
        self.matlab_configuration = MatlabConfiguration(configuration_data=configuration_data['matlab'])
        self.dataset_saver_configuration = DatasetSaverConfiguration(configuration_data=configuration_data['saver'])


class NeuralNetworkConfigurationBuilder(ConfigurationBuilder):
    """Class to define a neural network configuration builder"""
    architecture_configuration: "ConfigurationComponent" = None
    compile_configuration: "ConfigurationComponent" = None

    def build(self, configuration_data: Dict[str, Any]) -> None:
        """Build a neural network configuration object with its corresponding configuration components"""
        self.architecture_configuration = ArchitectureModelConfiguration(configuration_data=configuration_data['architecture'])
        self.compile_configuration = CompileModelConfiguration(configuration_data=configuration_data['compile'])
