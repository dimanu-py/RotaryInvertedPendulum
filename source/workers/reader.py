import pandas as pd

from source.helpers.data_loader import LoadData
from source.env_settings import EnvSettings


env_vars = EnvSettings.get_settings()


class Reader:

    def __init__(self, data_loader: LoadData):
        self.data_loader = data_loader

    def run(self) -> pd.DataFrame:
        """
        Run the data reader.
        :return: dataframe containing the data
        """
        data = self.data_loader.load_file()
        return data


if __name__ == '__main__':

    loader = LoadData(folder_path=env_vars.DATASETS_PATH)

    data_reader = Reader(data_loader=loader)
    data_reader.run()
