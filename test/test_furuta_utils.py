import pytest

from furuta_utils import read_yaml_parameters
from furuta_utils import extract_extension


@pytest.fixture
def folder_path():
    """
    Fixture to configure the path for the yaml used it to tests
    """
    return '../test/mock_data'


@pytest.fixture
def yaml_file_name():
    """
    Fixture to configure the name of the yaml used it to tests
    """
    return 'test_parameters.yaml'


def test_read_yaml_parameters(folder_path, yaml_file_name):
    """
    Tests that a yaml is read correctly
    """
    parameters = read_yaml_parameters(folder_path=folder_path,
                                      yaml_file=yaml_file_name)

    assert parameters is not None
    assert isinstance(parameters, dict)


def test_extract_extension(yaml_file_name):
    """
    Tests that extension is extracted correctly from any file
    """
    extension = extract_extension(file_name=yaml_file_name)

    assert extension is not None
    assert isinstance(extension, str)
    assert extension == 'yaml'


if __name__ == '__main__':
    pytest.main()
    