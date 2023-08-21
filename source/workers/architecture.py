from abc import abstractmethod, ABC

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from source.helpers.configuration_builder import Configuration


class Architecture(ABC):

    @abstractmethod
    def create_architecture(self) -> "Architecture":
        pass


class FullyConnectedNetwork(Architecture):
    CONFIG_KEY = 'neural_network_model'

    def __init__(self, configuration: "Configuration") -> None:
        self.model = None
        # TODO: manage to get directly architecture_config and not access it via neural_network_model
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY).architecture_config

    def create_architecture(self) -> "Architecture":

        self.model = Sequential()

        self.model.add(Input(shape=(self.configuration.input_shape, )))

        number_layers = len(self.configuration.number_units)
        last_layer_index = number_layers - 1

        for layer_index, layer_units in enumerate(self.configuration.number_units):
            activation_function = self.configuration.activation_hidden_layers
            if layer_index == last_layer_index:
                activation_function = self.configuration.activation_output_layer
            self.model.add(Dense(units=layer_units,
                                 activation=activation_function))
        # TODO: check this return -> if I return self I keep the encapsulation inside FullyConnectedNetwork but I have
        #  access the model via architecture.model. On the other hand, if I return self.model I get the model directly
        #  but I lose the encapsulation
        return self
