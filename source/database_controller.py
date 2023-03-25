import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.automap import automap_base

from source.database_settings import DatabaseSettings


class DatabaseController:

    def __init__(self, create_db_connection=False):
        self.settings = DatabaseSettings.get_settings()

        if create_db_connection:
            self.database_url = self.get_database_url()
            self.engine = create_engine(self.database_url)
            metadata = MetaData()
            metadata.reflect(self.engine)
            self.base = automap_base(metadata=metadata)
            self.base.prepare()
            self.database_tables_name = [str(table) for table in metadata.sorted_tables]

    def get_database_url(self) -> str:
        engine = self.settings.DB_CONNECTION
        port = self.settings.DB_PORT
        host = self.settings.DB_HOST
        database = self.settings.DB_NAME
        username = self.settings.DB_USER
        password = self.settings.DB_PASSWORD

        database_url = f'{engine}://{username}:{password}@{host}:{port}/{database}'

        return database_url

    def insert_data_to_database(self, table_name: str, data: pd.DataFrame = None) -> None:

        conn = self.create_connection()
        cursor = self.create_cursor(conn=conn)

        columns = self.get_table_columns(table_name=table_name)

        insert_data_query = self.insert_query(table_name=table_name,
                                              table_columns=columns)

        chunk = data.to_numpy().tolist()
        try:
            cursor.executemany(insert_data_query, chunk)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f'Error inserting data: {e}')
        finally:
            cursor.close()
            conn.close()

    def create_connection(self):
        try:
            conn = self.engine.raw_connection()
            return conn

        except Exception as e:
            print(f'Error {e.args[1]}')

    @staticmethod
    def create_cursor(conn):
        try:
            cursor = conn.cursor()
            return cursor
        except Exception as e:
            print(f'Error {e.args[1]}')

    def get_table_columns(self, table_name: str) -> list[str]:
        try:
            table = self.get_table(table_name=table_name)
            columns = [column.name for column in table.__table__.columns if not column.primary_key]
            return columns
        except Exception as e:
            print(f'Error {e.args[1]}')

    def get_table(self, table_name: str):
        try:
            table = getattr(self.base.classes, table_name)
            return table
        except Exception as e:
            print(f'Error {e.args[1]}')

    @staticmethod
    def insert_query(table_name: str, table_columns: list[str]) -> str:
        query = f"INSERT INTO {table_name} ({', '.join(table_columns)}) VALUES ({', '.join(['%s'] * len(table_columns))})"
        return query

    def read_data(self, table_name: str) -> pd.DataFrame:
        columns = self.get_table_columns(table_name=table_name)
        conn = self.create_connection()
        query = self.read_query(table_name=table_name,
                                table_columns=columns,
                                db_name=self.settings.DB_NAME)

        data = pd.read_sql(sql=query,
                           con=conn)

        return data

    @staticmethod
    def read_query(table_name: str, table_columns: list[str], db_name: str) -> str:
        query = f"SELECT {' ,'.join(table_columns)} FROM {db_name}.{table_name};"
        return query

