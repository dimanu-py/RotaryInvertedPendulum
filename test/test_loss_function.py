import pytest
from unittest.mock import Mock

from workers.loss_function import (MeanAbsoluteErrorLossFunction,
                                   MeanSquaredErrorLossFunction,
                                   MeanSquaredLogarithmicErrorLossFunction)


@pytest.fixture
def mock_loss_function_configuration():
    return {
        'loss': ['mse'],
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_loss_function_configuration):
    mock_configuration = Mock()
    mock_configuration.loss = mock_loss_function_configuration['loss']
    return mock_configuration


def test_mean_squared_error(mock_configuration_object):
    loss_function = MeanSquaredErrorLossFunction(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanSquaredErrorLossFunction)


def test_mean_absolute_error(mock_configuration_object):
    loss_function = MeanAbsoluteErrorLossFunction(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanAbsoluteErrorLossFunction)


def test_mean_squared_logarithmic_error(mock_configuration_object):
    loss_function = MeanSquaredLogarithmicErrorLossFunction(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanSquaredLogarithmicErrorLossFunction)
