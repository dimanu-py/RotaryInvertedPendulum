import pandas as pd
import pytest

from source.data_interface import DataInterface


@pytest.fixture(scope='function')
def data_interface_controller():
    return DataInterface()


def test_read_matlab_file(data_interface_controller):
    data = data_interface_controller.read_matlab_file()

    expected_columns = ['time', 'set_point_rotary_arm', 'control_law', 'position_rotary_arm', 'position_pendulum_wrapped', 'speed_rotary_arm', 'speed_pendulum', 'position_pendulum']

    assert not data.empty
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == expected_columns


@pytest.mark.parametrize('table_name', ['tests'])
def test_read_data_from_database(data_interface_controller, table_name):
    data = data_interface_controller.read_data_from_database(table_name=table_name)

    expected_columns = ['time', 'set_point_rotary_arm', 'control_law', 'position_rotary_arm', 'position_pendulum_wrapped', 'speed_rotary_arm', 'speed_pendulum', 'position_pendulum']

    assert not data.empty
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == expected_columns


@pytest.mark.parametrize('table_name, chunk_size', [('tests', 500_000)])
def test_insert_data_to_database(data_interface_controller, table_name, chunk_size):
    data = data_interface_controller.read_matlab_file()[0: chunk_size]
    data_interface_controller.insert_data_to_database(table_name=table_name,
                                                      data=data)
