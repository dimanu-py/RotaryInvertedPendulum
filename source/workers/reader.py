import pandas as pd

from source.helpers.data_loader import LoadData
from source.env_settings import EnvSettings
from source.helpers.furuta_utils import read_yaml_parameters


env_vars = EnvSettings.get_settings()


class DataReader:

    def __init__(self, data_loader: LoadData):
        self.data_loader = data_loader

        configuration_params = read_yaml_parameters().get('data_reader')
        self.file_name = configuration_params['file_name']
        self.extension = configuration_params['extension']
        self.columns_to_read = configuration_params['columns']

    def run(self) -> pd.DataFrame:
        """
        Run the data reader.
        :return: dataframe containing the data
        """
        raw_data = self.load_data(file_name=self.file_name,
                                  extension=self.extension)

        data = self.select_columns_to_read(data=raw_data,
                                           columns=self.columns_to_read)

        return data

    def load_data(self, file_name: str, extension: str) -> pd.DataFrame:
        """
        Load raw_data from a file.
        :return: dataframe containing the raw_data
        """
        raw_data = self.data_loader.load_file(file_name=file_name,
                                              extension=extension)
        return raw_data

    @staticmethod
    def select_columns_to_read(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Select columns to read from a file.
        :param data: dataframe containing the data
        :param columns: list of columns to read
        :return: dataframe containing the data
        """
        selected_data = data[columns]
        return selected_data


if __name__ == '__main__':

    loader = LoadData(folder_path=env_vars.DATASETS_PATH)

    data_reader = DataReader(data_loader=loader)
    data_reader.run()
