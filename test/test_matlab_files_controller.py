import pytest
import pandas as pd

from source.matlab_files_controller import MatlabFilesController


@pytest.fixture(scope='function')
def matlab_controller_object():
    return MatlabFilesController()


def test_read_parameters(matlab_controller_object):
    parameters = matlab_controller_object.read_parameters()
    assert isinstance(parameters, dict)


def test_signals_data(matlab_controller_object):
    data = matlab_controller_object.get_signals_data()
    assert isinstance(data, pd.DataFrame)


def test_load_matlab_file(matlab_controller_object):
    mat_data = matlab_controller_object.load_matlab_file()
    assert isinstance(mat_data, dict)
