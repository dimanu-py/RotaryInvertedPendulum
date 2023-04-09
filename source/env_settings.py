import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class EnvSettings:
    MATLAB_PATH: str
    DATASETS_PATH: str

    @classmethod
    def get_settings(cls) -> "EnvSettings":
        """
        Loads environment variables
        :return: instance of EnvSettings
        """
        load_dotenv()

        return cls(MATLAB_PATH=os.getenv('MATLAB_PATH'),
                   DATASETS_PATH=os.getenv('DATASETS_PATH'))
