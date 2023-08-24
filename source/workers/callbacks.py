from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint,
                             ReduceLROnPlateau,
                             TensorBoard)
from abc import ABC, abstractmethod
from source.helpers.configuration_builder import Configuration
from typing import List


class CallbacksFactory:

    @staticmethod
    def get_callbacks(configuration: "Configuration") -> List["Callbacks"]:
        callbacks_types = [callback_type for callback_type in configuration.callbacks.keys()]
        callbacks_classes = {'early_stopping': EarlyStoppingCallback,
                             'model_checkpoint': CheckpointCallback,
                             'reduce_lr': ReduceLearningRateCallback,
                             'tensor_board': TensorBoardCallback}

        try:
            callbacks = [callbacks_classes[callback](configuration.callbacks[callback]) for callback in callbacks_types]
            return [callback.callbacks for callback in callbacks]
        except KeyError:
            print(f'Callback {callbacks_types} not implemented yet')


class Callbacks(ABC):
    def __init__(self, configuration):
        self.configuration = configuration
        self.callbacks = self.configure_callback()

    @abstractmethod
    def configure_callback(self):
        pass


class EarlyStoppingCallback(Callbacks):

    def configure_callback(self):
        early_stopping = EarlyStopping(monitor=self.configuration['monitored_val'],
                                       patience=self.configuration['patience'],
                                       restore_best_weights=self.configuration['restore_best_weights'],
                                       verbose=self.configuration['verbose'])
        return early_stopping


class CheckpointCallback(Callbacks):

    def configure_callback(self):
        checkpoint = ModelCheckpoint(filepath=self.configuration['path'],
                                     monitor=self.configuration['monitored_val'],
                                     save_best_only=self.configuration['save_best'])
        return checkpoint


class ReduceLearningRateCallback(Callbacks):

    def configure_callback(self):
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor=self.configuration['monitored_val'],
                                                 factor=self.configuration['reduce_factor'],
                                                 patience=self.configuration['patience'],
                                                 verbose=self.configuration['verbose'])
        return reduce_lr_on_plateau


class TensorBoardCallback(Callbacks):

    def configure_callback(self):
        tensor_board = TensorBoard(log_dir=self.configuration['log_dir'],
                                   write_graph=self.configuration['write_graph'],
                                   write_images=self.configuration['write_images'])
        return tensor_board
