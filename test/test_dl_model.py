from source.dl_models.dl_model_creator import FullyConnectedNetwork
from source.helpers.configuration_reader import NeuralNetworkConfiguration
from source.env_settings import EnvSettings

env_vars = EnvSettings.get_env_vars()


def test_create_neural_network():
    configuration = NeuralNetworkConfiguration(configuration_path=env_vars.PARAMS_PATH)
    model_creator = FullyConnectedNetwork()

    model_creator.create_model(configuration=configuration)

    assert model_creator.model is not None
    assert isinstance(model_creator, FullyConnectedNetwork)
