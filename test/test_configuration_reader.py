from source.helpers.configuration_builder import Configuration, RawDatasetConfigurationBuilder
from source.env_settings import EnvSettings

env_vars = EnvSettings.get_env_vars()


def test_create_neural_network():
    configuration = Configuration(builder=RawDatasetConfigurationBuilder())
    raw_dataset_configuration = configuration.construct(env_vars.PARAMS_PATH, 'raw_dataset')
    assert raw_dataset_configuration.matlab_configuration is not None
