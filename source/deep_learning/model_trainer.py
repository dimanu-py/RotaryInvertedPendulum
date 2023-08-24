from typing import Tuple

import pandas as pd

from source.deep_learning.model_creator import DLModel
from source.helpers.configuration_builder import Configuration
from source.workers.splitter import DataSplitter
from source.workers.callbacks import CallbacksFactory


class Trainer:
    CONFIG_KEY = 'learning'

    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def train(self, model: "DLModel", dataset: pd.DataFrame) -> None:
        train_data, train_target, validation_data, validation_target, _, _ = self.split_datasets(dataset)
        callbacks = self.get_callbacks()
        history = model.model.fit(train_data.values,
                                  train_target.values,
                                  epochs=self.configuration.training_config.epochs,
                                  batch_size=self.configuration.training_config.batch_size,
                                  validation_data=(validation_data.values, validation_target.values),
                                  callbacks=callbacks,
                                  verbose=self.configuration.training_config.verbose)

    def split_datasets(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        data_splitter = DataSplitter(configuration=self.configuration.training_config)
        train_data, train_target, validation_data, validation_target, test_data, test_target = data_splitter.split_data(dataset=dataset)
        return train_data, train_target, validation_data, validation_target, test_data, test_target

    def get_callbacks(self):
        callbacks = CallbacksFactory.get_callbacks(self.configuration.callbacks_config)
        return callbacks


