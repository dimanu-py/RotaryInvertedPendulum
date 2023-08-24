import pandas as pd

from source.helpers.matlab_data_converter import MatlabDataConverter
from source.helpers.data_saver import SaveParquet
from source.helpers.shuffle_data import ShuffleTimeSeries
from source.helpers.configuration_builder import Configuration


class DatasetCreator:
    """
    Creates the training dataset gathering data from matlab file and saves it as parquet file.
    """
    CONFIG_KEY = 'data'

    # TODO: be able to get data from a variety of files type, not only .mat files
    def __init__(self, configuration: "Configuration") -> None:
        self.matlab_converter = MatlabDataConverter()
        self.shuffler = ShuffleTimeSeries()
        self.data_saver = SaveParquet()
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def create_training_dataset(self) -> pd.DataFrame:
        """
        Encapsulates the process of getting the raw dataset, shuffling the data using sliding window approach
        and saves the dataset in the dataset folder with parquet file.
        """
        raw_dataset = self._create_raw_dataset()
        # TODO: create logic to shuffle or not the data depending on the parameters yaml
        shuffled_dataset = self._shuffle_data(data=raw_dataset)
        return shuffled_dataset

    def _create_raw_dataset(self) -> pd.DataFrame:
        """
        Gets data from matlab files converted into dataframe.
        """
        matlab_configuration = self.configuration.matlab_config
        matlab_data = self.matlab_converter.transform_matlab_to_dataframe(data_file_name=matlab_configuration.file_name,
                                                                          mat_data_name=matlab_configuration.data,
                                                                          data_source=matlab_configuration.source,
                                                                          columns_name=matlab_configuration.columns)
        return matlab_data

    def _shuffle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        dataset_configuration = self.configuration.dataset_config
        shuffled_data = self.shuffler.shuffle_data(data=data,
                                                   window_length=dataset_configuration.window_length)
        return shuffled_data

    def save_dataset(self, data: pd.DataFrame) -> None:
        dataset_configuration = self.configuration.dataset_config
        self.data_saver.save_file(dataframe=data,
                                  folder_to_save=dataset_configuration.folder_to_save,
                                  save_file_name=dataset_configuration.dataset_name)
