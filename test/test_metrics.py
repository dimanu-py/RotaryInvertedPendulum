import pytest
from unittest.mock import Mock

from workers.metrics import (MeanAbsoluteErrorMetric,
                             MeanSquaredErrorMetric,
                             RootMeanSquaredErrorMetric,
                             MeanAbsolutePercentageErrorMetric,
                             MeanSquaredLogarithmicErrorMetric,
                             R2Metric)


@pytest.fixture
def mock_metrics_configuration():
    return {
        'metrics': ['mse', 'mae', 'rmse', 'mape', 'msle', 'r2'],
    }


@pytest.fixture(scope='function')
def mock_configuration_object(mock_metrics_configuration):
    mock_configuration = Mock()
    mock_configuration.metrics = mock_metrics_configuration['metrics']
    return mock_configuration


def test_mean_absolute_error(mock_configuration_object):
    metrics = MeanAbsoluteErrorMetric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanAbsoluteErrorMetric)


def test_mean_squared_error(mock_configuration_object):
    metrics = MeanSquaredErrorMetric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanSquaredErrorMetric)


def test_root_mean_squared_error(mock_configuration_object):
    metrics = RootMeanSquaredErrorMetric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, RootMeanSquaredErrorMetric)


def test_mean_absolute_percentage_error(mock_configuration_object):
    metrics = MeanAbsolutePercentageErrorMetric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanAbsolutePercentageErrorMetric)


def test_mean_squared_logarithmic_error(mock_configuration_object):
    metrics = MeanSquaredLogarithmicErrorMetric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, MeanSquaredLogarithmicErrorMetric)


def test_r2_score(mock_configuration_object):
    metrics = R2Metric(mock_configuration_object)
    assert metrics is not None
    assert isinstance(metrics, R2Metric)
