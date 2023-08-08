import abc

from source.furuta_utils import read_yaml_parameters


class ConfigurationFactory(abc.ABC):
    """Abstract class to make sure all configuration factory classes have the same method"""
    _registry = {}

    @classmethod
    def register_factory(cls, factory_type: str, factory: "ConfigurationFactory") -> None:
        cls._registry[factory_type] = factory

    @classmethod
    def create_factory(cls, factory_type: str) -> "ConfigurationFactory":
        factory = cls._registry.get(factory_type)
        try:
            return factory()
        except ValueError:
            print(f'Invalid factory type: {factory_type}')

    @abc.abstractmethod
    def create_configuration(self, configuration_data: dict) -> None:
        pass


class RawDatasetFactory(ConfigurationFactory):
    """Factory class for creating raw dataset configuration objects"""
    def __init__(self) -> None:
        self.register_factory(factory_type='raw_dataset', factory=self)

    def create_configuration(self, configuration_data: dict) -> ("MatlabConfiguration", "DatasetSaverConfiguration"):
        matlab_configuration = self.create_matlab_configuration(configuration_data=configuration_data['matlab'])
        dataset_saver_configuration = self.create_dataset_saver_configuration(configuration_data=configuration_data['saver'])
        return matlab_configuration, dataset_saver_configuration

    @staticmethod
    def create_matlab_configuration(configuration_data: dict) -> "MatlabConfiguration":
        return MatlabConfiguration(matlab_config=configuration_data)

    @staticmethod
    def create_dataset_saver_configuration(configuration_data: dict) -> "DatasetSaverConfiguration":
        return DatasetSaverConfiguration(saver_config=configuration_data)


class NeuralNetworkFactory(ConfigurationFactory):
    """Factory class for creating neural network configuration objects"""
    def __init__(self) -> None:
        self.register_factory(factory_type='neural_network', factory=self)

    def create_configuration(self, configuration_data: dict) -> ("BuildModelConfiguration", "CompileModelConfiguration"):
        build_configuration = self.create_build_configuration(configuration_data=configuration_data['build'])
        compile_configuration = self.create_compile_configuration(configuration_data=configuration_data['compile'])
        return build_configuration, compile_configuration

    @staticmethod
    def create_build_configuration(configuration_data: dict) -> "BuildModelConfiguration":
        return BuildModelConfiguration(build_config=configuration_data)

    @staticmethod
    def create_compile_configuration(configuration_data: dict) -> "CompileModelConfiguration":
        return CompileModelConfiguration(compile_config=configuration_data)


class Configuration(abc.ABC):
    """Abstract class to make sure all configuration classes have the same method"""
    @abc.abstractmethod
    def create_configuration(self, configuration_path: str) -> None:
        pass


class MatlabConfiguration:
    """Concrete product class for a matlab configuration object"""
    def __init__(self, matlab_config: dict) -> None:
        self.file_name = matlab_config.get('file_name')
        self.data = matlab_config.get('data')
        self.columns = matlab_config.get('columns')


class DatasetSaverConfiguration:

    def __init__(self, saver_config: dict) -> None:
        self.dataset_name = saver_config.get('dataset_name')


class RawDatasetConfiguration(Configuration):
    """Class that stores the configuration for creating a raw dataset"""
    matlab_config: MatlabConfiguration
    dataset_saver_config: DatasetSaverConfiguration
    CONFIG_KEY = 'raw_dataset'

    def __init__(self, config_factory: "ConfigurationFactory") -> None:
        self.config_factory = config_factory
        self.matlab_config = None
        self.dataset_saver_config = None

    def create_configuration(self, configuration_path: str) -> None:
        configuration = read_yaml_parameters(configuration_path).get(self.CONFIG_KEY)
        self.matlab_config, self.dataset_saver_config = self.config_factory.create_configuration(configuration_data=configuration)


class BuildModelConfiguration:

    def __init__(self, build_config: dict) -> None:
        self.type = build_config.get('model_type')
        self.input_shape = build_config.get('input_shape')
        self.number_units = build_config.get('number_units')
        self.activation_hidden_layers = build_config.get('activation_hidden_layers')
        self.activation_output_layer = build_config.get('activation_output_layer')


class CompileModelConfiguration:

    def __init__(self, compile_config: dict) -> None:
        self.loss = compile_config.get('loss')
        self.metrics = compile_config.get('metrics')
        self.optimizer = compile_config.get('optimizer')
        self.learning_rate = compile_config.get('learning_rate')


class NeuralNetworkConfiguration:
    build_config: BuildModelConfiguration
    compile_config: CompileModelConfiguration
    CONFIG_KEY = 'neural_network_model'

    def __init__(self, config_factory: "ConfigurationFactory") -> None:
        self.config_factory = config_factory
        self.build_config = None
        self.compile_config = None

    def create_configuration(self, configuration_path: str) -> None:
        configuration = read_yaml_parameters(configuration_path).get(self.CONFIG_KEY)
        self.build_config, self.compile_config = self.config_factory.create_configuration(configuration_data=configuration)
