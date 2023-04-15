from sklearn.model_selection import train_test_split
import pandas as pd

from source.helpers.furuta_utils import read_yaml_parameters

from source.workers.reader import DataReader
from source.helpers.data_loader import LoadData
from source.env_settings import EnvSettings

env_vars = EnvSettings.get_settings()


class DataSplitter:
    """
    Class to split data into train, validation and test data
    """
    def __init__(self, dataset: pd.DataFrame):
        self.raw_data = dataset

        configuration_params = read_yaml_parameters().get('splitter')
        self.target_column = configuration_params['target']
        self.shuffle_data = configuration_params['shuffle']
        self.test_split_ratio = configuration_params['test_size']
        self.validation_split_ratio = configuration_params['validation_size']

        self.data, self.target = self.separate_target(target=self.target_column)

    def split_data(self) -> tuple:
        """
        Main method. Runs the data splitter.
        :return: tuple containing the train, validation and test data
        """
        train_data, test_data, train_target, test_target = self.split_train_test(data=self.data,
                                                                                 target=self.target,
                                                                                 test_size=self.test_split_ratio,
                                                                                 shuffle_data=self.shuffle_data)

        train_data, validation_data, train_target, validation_target = self.split_train_validation(data=train_data,
                                                                                                   target=train_target,
                                                                                                   validation_size=self.validation_split_ratio,
                                                                                                   shuffle_data=self.shuffle_data)

        return (train_data, train_target), (validation_data, validation_target), (test_data, test_target)

    def separate_target(self, target: str) -> tuple:
        """
        Select target_column column from the dataset.
        :return:
        """
        data = self.raw_data.drop(columns=target)
        target = self.raw_data[target]
        return data, target

    @staticmethod
    def split_train_test(data: pd.DataFrame, target: pd.Dataframe, test_size: float, shuffle_data: bool) -> tuple:
        """
        Split train and test data.
        :param data: dataframe containing the data
        :param target: dataframe containing the target_column
        :param test_size: ratio of test data
        :param shuffle_data: boolean to shuffle data
        :return: tuple containing the train and test data
        """
        train_data, test_data, train_target, test_target = train_test_split(data,
                                                                            target,
                                                                            test_size=test_size,
                                                                            shuffle=shuffle_data)
        return train_data, test_data, train_target, test_target

    @staticmethod
    def split_train_validation(data: pd.DataFrame, target: pd.DataFrame, validation_size: float, shuffle_data: bool) -> tuple:
        """
        Split train and validation data.
        :param data: dataframe containing the data
        :param target: dataframe containing the target_column
        :param validation_size: ratio of validation data
        :param shuffle_data: boolean to shuffle data
        :return: tuple containing the train and validation data
        """
        train_data, validation_data, train_target, validation_target = train_test_split(data,
                                                                                        target,
                                                                                        test_size=validation_size,
                                                                                        shuffle=shuffle_data)
        return train_data, validation_data, train_target, validation_target


if __name__ == '__main__':

    loader = LoadData(folder_path=env_vars.DATASETS_PATH)
    reader = DataReader(data_loader=loader)
    raw_dataset = reader.read_data()
    splitter = DataSplitter(dataset=raw_dataset)

    (train_data, train_target), (validation_data, validation_target), (test_data, test_target) = splitter.split_data()
