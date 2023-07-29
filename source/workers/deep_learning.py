from source.helpers.loader_factory import LoaderFactory
from source.helpers.matlab_files_controller import MatlabFilesController
from source.helpers.saver_factory import SaverFactory

from source.workers.inserter import DataInserter
from source.workers.reader import DataReader
from source.workers.splitter import DataSplitter

from source.env_settings import EnvSettings


env_vars = EnvSettings.get_settings()


class DeepLearning:

    def __init__(self):
        self.matlab_controller = MatlabFilesController(matlab_folder=env_vars.MATLAB_PATH)
        self.dataset_saver = SaverFactory(folder_path=env_vars.DATASETS_PATH)
        self.data_inserter = DataInserter(matlab_controller=self.matlab_controller,
                                          file_saver=self.dataset_saver)

        # if insert_data:
        #     self.data_inserter.create_dataset()

        self.data_loader = LoaderFactory(folder_path=env_vars.DATASETS_PATH)
        self.data_reader = DataReader(data_loader=self.data_loader)

        self.raw_dataset = self.data_reader.read_data()

    def add_raw_dataset(self):
        ...


if __name__ == '__main__':

    deep_learning = DeepLearning()
    print(deep_learning.raw_dataset.head())
