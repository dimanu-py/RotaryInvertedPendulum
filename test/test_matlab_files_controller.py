import pytest
import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController
from source.furuta_utils import read_yaml_parameters


@pytest.fixture
def matlab_file_path():
    path = '../test/mock_data'
    return path


@pytest.fixture
def mock_yaml_params(matlab_file_path):
    parameters = read_yaml_parameters(folder_path=matlab_file_path,
                                      yaml_file='matlab_file_controller_test.yaml')
    return parameters


@pytest.fixture(scope='function')
def matlab_controller_object(matlab_file_path):
    return MatlabFilesController(matlab_folder=matlab_file_path)


def test_load_matlab_file(matlab_controller_object, mock_yaml_params):
    mat_data = matlab_controller_object._load_matlab_file(mat_file_name=mock_yaml_params['file_name'])
    assert isinstance(mat_data, dict)


def test_get_signals_data(matlab_controller_object, mock_yaml_params):
    data = matlab_controller_object.transform_matlab_to_dataframe(data_file_name=mock_yaml_params['file_name'],
                                                                  mat_data_name=mock_yaml_params['data_name'],
                                                                  columns_name=mock_yaml_params['columns'])

    assert not data.empty
    assert isinstance(data, pd.DataFrame)


