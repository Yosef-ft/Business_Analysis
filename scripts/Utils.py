import os
import pandas as pd 
from sqlalchemy import create_engine


class DatabaseConn:
    '''
    This class is for creating connection to the database and fetching data
    '''
    def __init__(self):
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_USER = os.getenv('DB_USER')      
        DB_NAME = os.getenv('DB_NAME')      
        DB_PASSWORD = os.getenv('DB_PASSWORD')      


        self.engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')

    def load_data_from_database(self) -> pd.DataFrame:
        '''
        This funtions creates a connection to the database and retrieves data from the database

        Returns:
        pd.DataFrame
        '''    
        with self.engine.connect() as conn:
            query = 'select * from xdr_data'
            df = pd.read_sql_query(query, conn)

        return df