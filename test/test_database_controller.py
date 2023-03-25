import pandas as pd
import pytest

from source.database_controller import DatabaseController


@pytest.fixture(scope='function')
def database_controller():
    return DatabaseController(create_db_connection=True)


def test_create_connection(database_controller):
    """
    Asserts that connection is established correctly
    """
    conn = database_controller.create_connection()
    assert conn is not None


def test_create_cursor(database_controller):
    """
    Asserts that cursor is created correctly
    """
    conn = database_controller.create_connection()
    cursor = database_controller.create_cursor(conn=conn)
    assert cursor is not None


def test_get_table_columns(database_controller):
    """
    Asserts table columns for each table in the database are retrieved correctly and as a list
    """
    database_tables = database_controller.database_tables_name
    for table_name in database_tables:
        table_columns = database_controller.get_table_columns(table_name=table_name)
        assert table_columns is not None
        assert isinstance(table_columns, list)


def test_get_table(database_controller):
    """
    Asserts each table of the database is access
    """
    database_tables = database_controller.database_tables_name
    for table_name in database_tables:
        table = database_controller.get_table(table_name=table_name)
        assert table is not None


@pytest.mark.parametrize('table_name, table_columns',
                         [('pruebas', ['time', 'set_point_rotary_arm', 'control_law', 'position_rotary_arm', 'position_pendulum_wrapped', 'speed_rotary_arm', 'speed_pendulum', 'position_pendulum'])])
def test_insert_query(database_controller, table_name, table_columns):
    expected_query = f"INSERT INTO {table_name} ({', '.join(table_columns)}) VALUES ({', '.join(['%s'] * len(table_columns))})"
    columns = database_controller.get_table_columns(table_name=table_name)

    query = database_controller.insert_query(table_name=table_name, table_columns=columns)

    assert query is not None
    assert query == expected_query


@pytest.mark.parametrize('table_name', ['pruebas'])
def test_read_data(database_controller, table_name):
    data = database_controller.read_data(table_name=table_name)
    columns_name = database_controller.get_table_columns(table_name=table_name)

    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert data.columns.all == columns_name
