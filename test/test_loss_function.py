import pytest
from unittest.mock import Mock

from source.deep_learning.loss_function import (LossFunction,
                                                MeanAbsoluteError,
                                                MeanSquaredError,
                                                MeanSquaredLogarithmicError)


@pytest.fixture
def mock_loss_function_configuration():
    return {
        'loss': 'mse',
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_loss_function_configuration):
    mock_configuration = Mock()
    mock_configuration.loss_function_config.loss = mock_loss_function_configuration['loss']
    return mock_configuration


def test_mean_squared_error(mock_configuration_object):
    loss_function = MeanSquaredError(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanSquaredError)


def test_mean_absolute_error(mock_configuration_object):
    loss_function = MeanAbsoluteError(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanAbsoluteError)


def test_mean_squared_logarithmic_error(mock_configuration_object):
    loss_function = MeanSquaredLogarithmicError(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, MeanSquaredLogarithmicError)


def test_loss_function(mock_configuration_object):
    loss_function = LossFunction(mock_configuration_object)
    assert loss_function is not None
    assert isinstance(loss_function, LossFunction)
    assert loss_function.get_loss_function() is not None
