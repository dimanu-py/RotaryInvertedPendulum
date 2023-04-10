import os.path

import pytest

from source.helpers.furuta_utils import read_yaml_parameters


@pytest.fixture
def default_yaml_path():
    current_file_path = os.path.dirname(__file__)
    return os.path.join(current_file_path, 'yaml_test.yaml')


def test_read_yaml_parameters(default_yaml_path):
    parameters = read_yaml_parameters(yaml_path=default_yaml_path)

    assert parameters is not None
    assert isinstance(parameters, dict)

