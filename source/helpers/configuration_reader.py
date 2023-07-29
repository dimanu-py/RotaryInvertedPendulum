from source.furuta_utils import read_yaml_parameters


class RawDatasetConfiguration:

    def __init__(self, configuration_path: str):
        configuration = read_yaml_parameters(configuration_path).get('raw_dataset')

        self.matlab_config = MatlabConfiguration(matlab_config=configuration.get('matlab'))
        self.dataset_saver_config = DatasetSaverConfiguration(saver_config=configuration.get('saver'))


class MatlabConfiguration:

    def __init__(self, matlab_config: dict) -> None:
        self.file_name = matlab_config.get('file_name')
        self.data = matlab_config.get('data')
        self.columns = matlab_config.get('columns')


class DatasetSaverConfiguration:

    def __init__(self, saver_config: dict) -> None:
        self.dataset_name = saver_config.get('dataset_name')


class NeuralNetworkConfiguration:

    def __init__(self, configuration_path: str) -> None:
        configuration = read_yaml_parameters(configuration_path).get('neural_network_model')

        self.build_config = BuildModelConfiguration(build_config=configuration.get('build'))
        self.compile_config = CompileModelConfiguration(compile_config=configuration.get('compile'))


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
