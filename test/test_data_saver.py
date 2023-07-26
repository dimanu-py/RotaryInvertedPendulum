import os

import pytest
import pandas as pd

from source.helpers.data_saver import SaveParquet


@pytest.fixture(scope='function')
def save_parquet_object():
    return SaveParquet(folder_path='../test/mock_data')


def test_check_folder_exists(save_parquet_object):
    save_parquet_object.check_folder_exists()
    assert True


def test_build_save_path(save_parquet_object):
    save_file_name = 'test'
    full_path = save_parquet_object.build_save_path(save_file_name)
    assert full_path == '../tests/mock_data\\test.parquet'


def test_save_file(save_parquet_object):
    dataframe = pd.DataFrame({'a': [1, 2, 3],
                              'b': [4, 5, 6]})
    save_file_name = 'data_saver_test'

    save_parquet_object.save_file(dataframe=dataframe,
                                  save_file_name=save_file_name)

    os.remove(f'../test/mock_data/{save_file_name}.parquet')