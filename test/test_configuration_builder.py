from source.helpers.configuration_builder import (Configuration,
                                                  RawDatasetConfigurationBuilder,
                                                  NeuralNetworkConfigurationBuilder)

import pytest


@pytest.fixture
def yaml_test_file_data():
    folder_path = "../test/mock_data"
    yaml_file_name = "test_parameters.yaml"
    return folder_path, yaml_file_name


@pytest.fixture
def mock_configuration_raw_dataset():
    return {
        'matlab': {
            'file_name': 'matlab_file_test.mat',
            'data': 'data',
            'columns': ['time', 'set_point_rotary_arm',
                        'control_law',
                        'position_rotary_arm',
                        'position_pendulum_wrapped',
                        'speed_rotary_arm',
                        'speed_pendulum',
                        'position_pendulum']
        },
        'saver': {
            'dataset_name': 'dataset_test.parquet'
        }
    }


@pytest.fixture
def mock_configuration_neural_network():
    return {
        'architecture': {
            'input_shape': 8,
            'output_shape': 1,
            'number_units': [8, 8, 8],
            'activation_hidden_layers': 'relu',
            'activation_output_layer': 'linear'
        },
        'compile': {
            'loss': 'mse',
            'metrics': ['mae', 'mse'],
            'optimizer': 'adam',
            'learning_rate': 0.001
        }
    }


def test_raw_dataset_configuration_builder(mock_configuration_raw_dataset):
    builder = RawDatasetConfigurationBuilder()
    builder.build(configuration_data=mock_configuration_raw_dataset)

    assert isinstance(builder, RawDatasetConfigurationBuilder)
    assert builder.matlab_configuration is not None
    assert builder.dataset_saver_configuration is not None


def test_neural_network_configuration_builder(mock_configuration_neural_network):
    builder = NeuralNetworkConfigurationBuilder()
    builder.build(configuration_data=mock_configuration_neural_network)

    assert isinstance(builder, NeuralNetworkConfigurationBuilder)
    assert builder.architecture_configuration is not None
    assert builder.compile_configuration is not None


@pytest.mark.parametrize("data_key", ["raw_dataset"])
def test_create_raw_dataset_configuration(yaml_test_file_data, data_key):
    builder = RawDatasetConfigurationBuilder()
    configuration = Configuration(builder=builder)
    raw_dataset_configuration = configuration.construct(data_key,
                                                        folder_path=yaml_test_file_data[0],
                                                        yaml_file=yaml_test_file_data[1])

    assert isinstance(raw_dataset_configuration, Configuration)
    assert raw_dataset_configuration.matlab_configuration is not None
    assert raw_dataset_configuration.dataset_saver_configuration is not None


@pytest.mark.parametrize("data_key", ["neural_network_model"])
def test_create_neural_network_configuration(yaml_test_file_data, data_key):
    builder = NeuralNetworkConfigurationBuilder()
    configuration = Configuration(builder=builder)
    neural_network_configuration = configuration.construct(data_key,
                                                           folder_path=yaml_test_file_data[0],
                                                           yaml_file=yaml_test_file_data[1])

    assert isinstance(neural_network_configuration, Configuration)
    assert neural_network_configuration.architecture_configuration is not None
    assert neural_network_configuration.compile_configuration is not None
