import pandas as pd
import pytest

from source.data_interface import DataInterface


@pytest.fixture(scope='function')
def data_interface_controller():
    return DataInterface()


def test_read_matlab_file(data_interface_controller):
    data = data_interface_controller.read_matlab_file()

    expected_columns = ['time', 'set_point_rotary_arm', 'control_law', 'position_rotary_arm', 'position_pendulum_wrapped', 'speed_rotary_arm', 'speed_pendulum', 'position_pendulum']

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == expected_columns


