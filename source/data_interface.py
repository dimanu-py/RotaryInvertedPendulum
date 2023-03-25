from source.database_controller import DatabaseController
from source.matlab_files_controller import MatlabFilesController


class DataInterface:
    def __init__(self):
        self.matlab_files_controller = MatlabFilesController()
        self.database_controller = DatabaseController()

