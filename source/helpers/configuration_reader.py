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
