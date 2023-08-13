from sklearn.model_selection import train_test_split
import pandas as pd

from source.helpers.furuta_utils import read_yaml_parameters


# TODO: delete __main__ when inserting into deep_learning.py
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

        assert train_data.shape[0] > validation_data.shape[0]

        return train_data, train_target, validation_data, validation_target, test_data, test_target

    def separate_target(self, target: str) -> tuple:
        """
        Select target_column column from the dataset.
        :return:
        """
        data = self.raw_data.drop(columns=target)
        target = self.raw_data[target]
        return data, target

    @staticmethod
    def split_train_test(data: pd.DataFrame, target: pd.DataFrame, test_size: float, shuffle_data: bool) -> tuple:
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
    from source.deep_learning.reader import DataReader
    from source.helpers.data_loader import LoaderFactory
    from source.env_settings import EnvSettings

    env_vars = EnvSettings.get_settings()

    loader = LoaderFactory(folder_path=env_vars.DATASETS_PATH)
    reader = DataReader(data_loader=loader)
    raw_dataset = reader.read_data()
    splitter = DataSplitter(dataset=raw_dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data()

    print("Forma de X_train:", X_train.shape)
    print("Forma de y_train:", y_train.shape)
    print("Forma de X_val:", X_val.shape)
    print("Forma de y_val:", y_val.shape)
    print("Forma de X_test:", X_test.shape)
    print("Forma de y_test:", y_test.shape)
