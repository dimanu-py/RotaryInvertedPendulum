import tensorflow as tf
from abc import ABC, abstractmethod
from source.helpers.configuration_builder import Configuration
from typing import List


class CallbacksFactory:

    @staticmethod
    def get_callbacks(configuration: "Configuration") -> List["Callbacks"]:
        callbacks_type = configuration.callbacks
        callbacks_classes = {'early_stopping': EarlyStopping,
                             'model_checkpoint': Checkpoint,
                             'reduce_lr': ReduceLearningRate,
                             'tensor_board': TensorBoard}

        try:
            callbacks = [callbacks_classes[callback](configuration) for callback in callbacks_type]
            return [callback.callbacks for callback in callbacks]
        except KeyError:
            print(f'Callback {callbacks_type} not implemented yet')


class Callbacks(ABC):
    def __init__(self, configuration):
        self.configuration = configuration

    @abstractmethod
    def configure_callback(self):
        pass


class EarlyStopping(Callbacks):

    def configure_callback(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.configuration.callback_name,
                                                          patience=self.configuration.patience,
                                                          restore_best_weights=self.configuration.restore_best_weights,
                                                          verbose=self.configuration.verbose)
        return early_stopping


class Checkpoint(Callbacks):

    def configure_callback(self):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.configuration.checkpoint_path,
                                                        monitor=self.configuration.monitor,
                                                        save_best_only=self.configuration.save_best)
        return checkpoint


class ReduceLearningRate(Callbacks):

    def configure_callback(self):
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.configuration.monitor,
                                                                    factor=self.configuration.reduce_factor,
                                                                    patience=self.configuration.patience,
                                                                    verbose=self.configuration.verbose)
        return reduce_lr_on_plateau


class TensorBoard(Callbacks):

    def configure_callback(self):
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.configuration.log_dir,
                                                      write_graph=self.configuration.write_graph,
                                                      write_images=self.configuration.write_images)
        return tensor_board
