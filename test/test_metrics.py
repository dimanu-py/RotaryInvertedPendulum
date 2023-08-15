import pytest
from unittest.mock import Mock

from workers.metrics import (MeanAbsoluteError,
                             MeanSquaredError,
                             RootMeanSquaredError,
                             MeanAbsolutePercentageError,
                             MeanSquaredLogarithmicError,
                             R2Score,
                             Metrics)


@pytest.fixture
def mock_metrics_configuration():
    return {
        'metrics': ['mse', 'mae', 'rmse', 'mape', 'msle', 'r2'],
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_metrics_configuration):
    mock_configuration = Mock()
    mock_configuration.metrics_config.metrics = mock_metrics_configuration['metrics']
    return mock_configuration


def test_mean_absolute_error(mock_configuration_object):
    metrics = MeanAbsoluteError(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanAbsoluteError)


def test_mean_squared_error(mock_configuration_object):
    metrics = MeanSquaredError(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanSquaredError)


def test_root_mean_squared_error(mock_configuration_object):
    metrics = RootMeanSquaredError(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, RootMeanSquaredError)


def test_mean_absolute_percentage_error(mock_configuration_object):
    metrics = MeanAbsolutePercentageError(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanAbsolutePercentageError)


def test_mean_squared_logarithmic_error(mock_configuration_object):
    metrics = MeanSquaredLogarithmicError(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanSquaredLogarithmicError)


def test_r2_score(mock_configuration_object):
    metrics = R2Score(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, R2Score)


def test_metrics(mock_configuration_object):
    metrics = Metrics(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, Metrics)
    assert metrics.get_metrics() is not None
    assert len(metrics.get_metrics()) == len(mock_configuration_object.metrics_config.metrics)
