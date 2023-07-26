import pytest
import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.furuta_utils import read_yaml_parameters


@pytest.fixture
def matlab_file_path():
    path = '../test/mock_data'
    return path


@pytest.fixture
def configuration_yaml_path():
    path = '../test/mock_data/matlab_file_controller_yaml_test.yaml'
    return path


@pytest.fixture
def read_yaml_params(configuration_yaml_path):
    parameters = read_yaml_parameters(yaml_path=configuration_yaml_path)
    return parameters


@pytest.fixture(scope='function')
def matlab_controller_object(matlab_file_path):
    return MatlabFilesController(matlab_folder=matlab_file_path)


def test_load_matlab_file(matlab_controller_object, read_yaml_params):
    mat_data = matlab_controller_object._load_matlab_file(mat_file_name=read_yaml_params['file_name'])
    assert isinstance(mat_data, dict)


def test_get_signals_data(matlab_controller_object, read_yaml_params):
    data = matlab_controller_object.transform_matlab_to_dataframe(data_file_name=read_yaml_params['file_name'],
                                                                  mat_data_name=read_yaml_params['data_name'],
                                                                  columns_name=read_yaml_params['columns_name'])

    assert not data.empty
    assert isinstance(data, pd.DataFrame)


