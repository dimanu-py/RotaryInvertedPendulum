import os.path

import pytest

from source.helpers.furuta_utils import read_yaml_parameters
from source.helpers.furuta_utils import extract_extension


@pytest.fixture
def yaml_testing():
    """
    Fixture to configure the path for the yaml used it to tests
    """
    current_file_path = os.path.dirname(__file__)
    return os.path.join(current_file_path, 'yaml_test.yaml')


def test_read_yaml_parameters(yaml_testing):
    """
    Tests that a yaml is read correctly
    """
    parameters = read_yaml_parameters(yaml_path=yaml_testing)

    assert parameters is not None
    assert isinstance(parameters, dict)


def test_extract_extension(yaml_testing):
    """
    Tests that extension is extracted correctly from any file
    """
    extension = extract_extension(file_name=yaml_testing)

    assert extension is not None
    assert isinstance(extension, str)
    assert extension == 'yaml'
