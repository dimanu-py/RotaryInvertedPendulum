import pytest
from unittest.mock import Mock

from source.deep_learning.optimizer import (AdamOptimizer,
                                            SGDOptimizer,
                                            RMSPropOptimizer,
                                            Optimizer)


@pytest.fixture
def mock_optimizer_configuration():
    return {
        'optimizer_type': 'adam',
        'learning_rate': 0.001,
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_optimizer_configuration):
    mock_configuration = Mock()
    mock_configuration.optimizer_config.optimizer_type = mock_optimizer_configuration['optimizer_type']
    mock_configuration.optimizer_config.learning_rate = mock_optimizer_configuration['learning_rate']
    return mock_configuration


def test_adam_optimizer(mock_configuration_object):
    optimizer = AdamOptimizer(mock_configuration_object)
    assert optimizer is not None
    assert isinstance(optimizer, AdamOptimizer)


def test_sgd_optimizer(mock_configuration_object):
    optimizer = SGDOptimizer(mock_configuration_object)
    assert optimizer is not None
    assert isinstance(optimizer, SGDOptimizer)


def test_rmsprop_optimizer(mock_configuration_object):
    optimizer = RMSPropOptimizer(mock_configuration_object)
    assert optimizer is not None
    assert isinstance(optimizer, RMSPropOptimizer)


def test_optimizer(mock_configuration_object):
    optimizer = Optimizer(mock_configuration_object)
    assert optimizer is not None
    assert isinstance(optimizer, Optimizer)
    assert optimizer.optimizer is not None


