from source.workers.architecture import FullyConnectedNetwork

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
    mock_configuration.input_shape = mock_architecture_configuration['input_shape']
    mock_configuration.output_shape = mock_architecture_configuration['output_shape']
    mock_configuration.number_units = mock_architecture_configuration['number_units']
    mock_configuration.activation_hidden_layers = mock_architecture_configuration['activation_hidden_layers']
    mock_configuration.activation_output_layer = mock_architecture_configuration['activation_output_layer']
    return mock_configuration


def test_create_architecture(mock_configuration_object):
    architect = FullyConnectedNetwork(mock_configuration_object)
    model = architect.create_architecture()

    assert model is not None
    assert isinstance(model, FullyConnectedNetwork)
    assert model.model is not None


if __name__ == '__main__':
    pytest.main()
