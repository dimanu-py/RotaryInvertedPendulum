import abc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from source.helpers.configuration_reader import NeuralNetworkConfiguration


# class Models(abc.ABC):
#
#     @abc.abstractmethod
#     def create_model(self):
#         pass
#
#     @abc.abstractmethod
#     def compile_model(self):
#         pass
#
#     @abc.abstractmethod
#     def callbacks(self):
#         pass


class FullyConnectedNetwork:

    def __init__(self) -> None:
        self.model = None

    def create_model(self, configuration: NeuralNetworkConfiguration) -> "FullyConnectedNetwork":
        build = configuration.build_config

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

    def compile_model(self):
        pass

    def callbacks(self):
        pass


class ModelsFactory:

    models = {'dense': FullyConnectedNetwork}

    def __init__(self, architecture: str):
        self.architecture = architecture

    def create_model(self):
        selected_model = ModelsFactory.models.get(self.architecture)

        if not selected_model:
            raise ValueError('Architecture not supported')

        model_instance = selected_model()
        model = model_instance.create_model()

        return model

