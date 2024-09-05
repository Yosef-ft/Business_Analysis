import os
import pandas as pd 
from scipy.stats import zscore
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
    

class DataUtils:
    def __init__(self,data: pd.DataFrame):
        self.data = data

    def data_info(self):
        '''
        Provides information about the data: 
            * provides the percentage of missing values
            * The number of missing values for each column
            * the data types of the missing values
        '''
        
        missing_values = self.data.isna().sum()
        missing_percent = self.data.isna().mean() * 100 
        data_types = self.data.dtypes

        info_df = pd.DataFrame({
            "Missing values" : missing_values, 
            "Missing Percentage" : missing_percent, 
            "Dtypes" : data_types
        })

        info_df = info_df[missing_percent > 0]

        max_na_col = info_df.loc[info_df['Missing values'] == info_df['Missing values'].max()].index
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)

        print(f'The data contains {self.data.shape[0]} number of rows and {self.data.shape[1]} number of columns.\n'
            f'The data has {info_df.shape[0]} number of missing columns\n'
            f'The data with the maximum number of missing columns is {max_na_col}\n\n'
            f'The data column containing more than 50% missing values are:')
        print(*more_than_half_na, sep='\n')
            
        

        return info_df