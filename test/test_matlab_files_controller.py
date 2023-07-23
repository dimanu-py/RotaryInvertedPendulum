import pytest
import pandas as pd

from source.helpers.matlab_files_controller import MatlabFilesController


@pytest.fixture(scope='function')
def matlab_controller_object():
    return MatlabFilesController()


def test_get_signals_data(matlab_controller_object):
    data = matlab_controller_object._transform_matlab_to_dataframe()

    assert not data.empty
    assert isinstance(data, pd.DataFrame)


@pytest.mark.parametrize('mat_file_path', [r'C:\PROGRAMACION\PENDULO INVERTIDO\Pendulo Invertido Diego\Matlab-Furuta-Pendulum\Simulink\dataAcquisition\synthetic_data.mat'])
def test_load_matlab_file(matlab_controller_object, mat_file_path):
    mat_data = matlab_controller_object._load_matlab_file(,
    assert isinstance(mat_data, dict)
