import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class DatabaseSettings:
    DB_CONNECTION: str
    DB_PORT: str
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    @staticmethod
    def get_settings():
        load_dotenv()

        settings = DatabaseSettings(DB_CONNECTION=os.getenv('DB_CONNECTION'),
                                    DB_PORT=os.getenv('DB_PORT'),
                                    DB_HOST=os.getenv('DB_HOST'),
                                    DB_USER=os.getenv('DB_USER'),
                                    DB_PASSWORD=os.getenv('DB_PASSWORD'),
                                    DB_NAME=os.getenv('DB_NAME'))

        return settings
