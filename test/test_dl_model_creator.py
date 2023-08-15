from deep_learning.dl_model_creator import FullyConnectedNetwork

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_architecture_configuration():
    return {
        'input_shape': 8,
        'output_shape': 1,
        'number_units': [8, 8, 8],
        'activation_hidden_layers': 'relu',
        'activation_output_layer': 'linear'
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_architecture_configuration):
    mock_configuration = Mock()
    mock_configuration.architecture_config.input_shape = mock_architecture_configuration['input_shape']
    mock_configuration.architecture_config.output_shape = mock_architecture_configuration['output_shape']
    mock_configuration.architecture_config.number_units = mock_architecture_configuration['number_units']
    mock_configuration.architecture_config.activation_hidden_layers = mock_architecture_configuration['activation_hidden_layers']
    mock_configuration.architecture_config.activation_output_layer = mock_architecture_configuration['activation_output_layer']
    return mock_configuration


def test_create_architecture(mock_configuration_object):
    model = FullyConnectedNetwork(mock_configuration_object)
    model.create_architecture()

    assert model is not None
    assert isinstance(model, FullyConnectedNetwork)
    assert model.model is not None


if __name__ == '__main__':
    pytest.main()
