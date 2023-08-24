import random
from abc import ABC, abstractmethod
import pandas as pd


class ShuffleData(ABC):

    @abstractmethod
    def shuffle_data(self, data: pd.DataFrame, window_length: int) -> pd.DataFrame:
        pass


class ShuffleTimeSeries(ShuffleData):

    def shuffle_data(self, data: pd.DataFrame, window_length: int) -> pd.DataFrame:
        number_windows = self.calculate_number_windows(data=data,
                                                       window_length=window_length)

        sliding_windows = self.extract_sliding_windows(data=data,
                                                       window_length=window_length,
                                                       number_windows=number_windows)

        random.shuffle(sliding_windows)
        shuffled_dataset = pd.concat(sliding_windows, ignore_index=True)
        return shuffled_dataset

    @staticmethod
    def extract_sliding_windows(data: pd.DataFrame, window_length: int, number_windows: int):
        sliding_windows = []
        for i in range(number_windows):
            first_index = i * window_length
            last_index = first_index + window_length
            window = data.iloc[first_index: last_index]
            sliding_windows.append(window)
        return sliding_windows

    @staticmethod
    def calculate_number_windows(data: pd.DataFrame, window_length: int) -> int:
        return len(data) // window_length + (len(data) % window_length != 0)
