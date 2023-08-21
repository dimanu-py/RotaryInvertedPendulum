from abc import ABC, abstractmethod


class DesignSystem(ABC):

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def create_model_architecture(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def set_callbacks(self):
        pass

    @abstractmethod
    def split_datasets(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass
