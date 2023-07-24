import os.path

import pytest

from source.helpers.furuta_utils import read_yaml_parameters
from source.helpers.furuta_utils import extract_extension


@pytest.fixture
def yaml_path():
    """
    Fixture to configure the path for the yaml used it to tests
    """
    return '../test/mock_data/read_parameter_yaml_test.yaml'


def test_read_yaml_parameters(yaml_path):
    """
    Tests that a yaml is read correctly
    """
    parameters = read_yaml_parameters(yaml_path=yaml_path)

    assert parameters is not None
    assert isinstance(parameters, dict)


def test_extract_extension(yaml_path):
    """
    Tests that extension is extracted correctly from any file
    """
    extension = extract_extension(file_name=yaml_path)

    assert extension is not None
    assert isinstance(extension, str)
    assert extension == 'yaml'
