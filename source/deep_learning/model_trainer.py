from source.helpers.configuration_builder import Configuration


class Trainer:
    CONFIG_KEY = 'training'

    def __init__(self, configuration: "Configuration"):
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def split_datasets(self):
        pass

    def shuffle_timeseries_data(self):
        pass

    def train(self):
        pass
