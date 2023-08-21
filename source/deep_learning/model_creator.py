from source.helpers.configuration_builder import Configuration


class Model:
    CONFIG_KEY = 'neural_network_model'

    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def architecture(self):
        pass

    def compile(self):
        pass
