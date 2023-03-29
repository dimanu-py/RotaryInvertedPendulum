import pandas as pd
import pytest

from source.data_reader import DataReader


@pytest.fixture(scope='function')
def data_reader_controller():
    return DataReader()


def test_read_matlab_file(data_reader_controller):
    with pytest.raises(NotImplementedError):
        data_reader_controller.read_matlab_file()


def test_insert_data_to_database(data_reader_controller):
    with pytest.raises(NotImplementedError):
        data_reader_controller.insert_data_to_database(data=pd.DataFrame(),
                                                       table_name='table')


@pytest.mark.parametrize('table_name', ['tests'])
def test_read_data_from_database(data_reader_controller, table_name):
    data = data_reader_controller.read_data_from_database(table_name=table_name)  # read all columns

    expected_columns = ['time', 'set_point_rotary_arm', 'control_law', 'position_rotary_arm',
                        'position_pendulum_wrapped', 'speed_rotary_arm', 'speed_pendulum', 'position_pendulum']

    assert not data.empty
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == expected_columns


@pytest.mark.parametrize('table_name', ['tests'])
def test_run_data_reader(data_reader_controller, table_name):
    data = data_reader_controller.run()

    assert not data.empty
    assert isinstance(data, pd.DataFrame)
