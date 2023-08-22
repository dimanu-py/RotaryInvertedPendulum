import pandas as pd

from source.helpers.matlab_data_converter import MatlabDataConverter
from source.helpers.data_saver import SaveParquet
from source.helpers.configuration_builder import Configuration


class DatasetCreator:
    """
    Creates the training dataset gathering data from matlab file and saves it as parquet file.
    """
    CONFIG_KEY = 'raw_dataset'

    # TODO: be able to get data from a variety of files type, not only .mat files
    def __init__(self, configuration: Configuration) -> None:
        self.matlab_converter = MatlabDataConverter()
        self.data_saver = SaveParquet()
        self.configuration = configuration.construct(data_key=self.CONFIG_KEY)

    def create_dataset(self) -> pd.DataFrame:
        """
        Main method. Gets data from matlab files converted into dataframe.
        """
        matlab_configuration = self.configuration.matlab_config
        matlab_data = self.get_data_from_matlab(matlab_configuration=matlab_configuration)
        return matlab_data

    def get_data_from_matlab(self, matlab_configuration) -> pd.DataFrame:
        """
        Get data from matlab file. Encapsulates call to MatlabConverter
        """
        data = self.matlab_converter.transform_matlab_to_dataframe(data_file_name=matlab_configuration.file_name,
                                                                   mat_data_name=matlab_configuration.data,
                                                                   data_source=matlab_configuration.source,
                                                                   columns_name=matlab_configuration.columns)
        return data

    def save_dataset(self, data: pd.DataFrame) -> None:
        saver_configuration = self.configuration.dataset_saver_config
        self.data_saver.save_file(dataframe=data,
                                  folder_to_save=saver_configuration.folder_to_save,
                                  save_file_name=saver_configuration.dataset_name)
