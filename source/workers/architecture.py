from abc import abstractmethod, ABC

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from source.helpers.configuration_builder import Configuration


class Architecture(ABC):

    @abstractmethod
    def create_architecture(self, configuration: "Configuration") -> "Architecture":
        pass


class FullyConnectedNetwork(Architecture):

    def __init__(self) -> None:
        self.model = None

    def create_architecture(self, configuration: "Configuration") -> "Architecture":

        self.model = Sequential()

        self.model.add(Input(shape=(configuration.input_shape, )))

        number_layers = len(configuration.number_units)
        last_layer_index = number_layers - 1
        for layer_index, layer_units in enumerate(configuration.number_units):
            activation_function = configuration.activation_hidden_layers
            if layer_index == last_layer_index:
                activation_function = configuration.activation_output_layer
            self.model.add(Dense(units=layer_units,
                                 activation=activation_function))
        return self
