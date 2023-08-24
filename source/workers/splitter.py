from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

from source.helpers.configuration_builder import Configuration


class DataSplitter:
    """
    Class to split data into train, validation and test data
    """
    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration

    def split_data(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Runs the data splitter.
        """
        data, target = self.separate_target(data=dataset,
                                            features=self.configuration.features,
                                            target=self.configuration.target)

        train_data, test_data, train_target, test_target = self.split_train_test(data=data,
                                                                                 target=target,
                                                                                 test_size=self.configuration.test_size)

        train_data, validation_data, train_target, validation_target = self.split_train_validation(data=train_data,
                                                                                                   target=train_target,
                                                                                                   validation_size=self.configuration.validation_size)

        assert train_data.shape[0] > validation_data.shape[0]

        return train_data, train_target, validation_data, validation_target, test_data, test_target

    @staticmethod
    def separate_target(data: pd.DataFrame, features: List[str], target: List[str]) -> Tuple[pd.DataFrame, ...]:
        """
        Select target_column column from the dataset.
        """
        dataset = data[features]
        training_target = data[target]
        return dataset, training_target

    @staticmethod
    def split_train_test(data: pd.DataFrame, target: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, ...]:
        """
        Split train and test data.
        """
        train_data, test_data, train_target, test_target = train_test_split(data,
                                                                            target,
                                                                            test_size=test_size)
        return train_data, test_data, train_target, test_target

    @staticmethod
    def split_train_validation(data: pd.DataFrame, target: pd.DataFrame, validation_size: float) -> Tuple[pd.DataFrame, ...]:
        """
        Split train and validation data.
        """
        train_data, validation_data, train_target, validation_target = train_test_split(data,
                                                                                        target,
                                                                                        test_size=validation_size)
        return train_data, validation_data, train_target, validation_target
