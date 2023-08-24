from typing import Tuple

import pandas as pd

from source.deep_learning.model_creator import DLModel
from source.helpers.configuration_builder import Configuration
from source.workers.splitter import DataSplitter


class Trainer:
    CONFIG_KEY = 'training'

    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def split_datasets(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        data_splitter = DataSplitter(configuration=self.configuration.training_config)
        train_data, train_target, validation_data, validation_target, test_data, test_target = data_splitter.split_data(dataset=dataset)
        return train_data, train_target, validation_data, validation_target, test_data, test_target

    def train(self, model: "DLModel", dataset: pd.DataFrame) -> None:
        train_data, train_target, validation_data, validation_target, test_data, test_target = self.split_datasets(dataset)

