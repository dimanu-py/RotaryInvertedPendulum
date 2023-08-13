from abc import abstractmethod

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from source.helpers.configuration_builder import Configuration


class Model:

    @abstractmethod
    def create_architecture(self) -> "Model":
        pass


class FullyConnectedNetwork(Model):

    def __init__(self, configuration: "Configuration") -> None:
        self.model = None
        self.configuration = configuration

    def create_architecture(self) -> "Model":
        build = self.configuration.architecture_config

        self.model = Sequential()

        self.model.add(Input(shape=(build.input_shape, )))

        number_layers = len(build.number_units)
        last_layer_index = number_layers - 1
        for layer_index, layer_units in enumerate(build.number_units):
            activation_function = build.activation_hidden_layers
            if layer_index == last_layer_index:
                activation_function = build.activation_output_layer
            self.model.add(Dense(units=layer_units,
                                 activation=activation_function))
        return self
