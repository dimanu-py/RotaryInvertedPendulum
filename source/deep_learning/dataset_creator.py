import pandas as pd

from source.helpers.matlab_data_converter import MatlabDataConverter
from source.helpers.configuration_builder import Configuration


class DatasetCreator:
    """
    Creates the training dataset gathering data from matlab file.
    """
    CONFIG_KEY = 'raw_dataset'

    # TODO: be able to get data from a variety of files type, not only .mat files
    def __init__(self, matlab_converter: MatlabDataConverter, configuration: Configuration) -> None:
        self.matlab_converter = matlab_converter
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
