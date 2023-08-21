from abc import ABC, abstractmethod


class DesignSystem(ABC):

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def training(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass
