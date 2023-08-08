from source.helpers.configuration_reader import RawDatasetConfiguration, RawDatasetFactory
from source.env_settings import EnvSettings

env_vars = EnvSettings.get_env_vars()


def test_create_neural_network():
    configuration = RawDatasetConfiguration(config_factory=RawDatasetFactory())
    configuration.create_configuration(configuration_path=env_vars.PARAMS_PATH)
