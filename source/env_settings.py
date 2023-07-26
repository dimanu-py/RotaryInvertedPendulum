import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class EnvSettings:
    """
    Class to store personal paths specified inside .env file
    """
    MATLAB_PATH: str
    PARAMS_PATH: str
    DATASETS_PATH: str

    @classmethod
    def get_env_vars(cls) -> "EnvSettings":
        """
        Loads environment variables
        """
        load_dotenv()

        return cls(MATLAB_PATH=os.getenv('MATLAB_PATH'),
                   PARAMS_PATH=os.getenv('PARAMETERS_PATH'),
                   DATASETS_PATH=os.getenv('DATASETS_PATH'))
