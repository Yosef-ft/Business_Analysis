import os
import pandas as pd 
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
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
    '''
    This class is used to clean, visualize and identify outliers from the dataset
    '''
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
        info_df = info_df.sort_values(by='Missing Percentage', ascending=False)

        max_na_col = list(info_df.loc[info_df['Missing values'] == info_df['Missing values'].max()].index)
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)

        print(f"**The data contains `{self.data.shape[0]}` rows and `{self.data.shape[1]}` columns.**\n\n"
            f"**The data has `{info_df.shape[0]}` missing columns.**\n\n"
            f"**The column with the maximum number of missing values is `{max_na_col}`.**\n\n"
            f"**Columns with more than 50% missing values are:**")

        # Print the list of columns with more than 50% missing values
        for column in more_than_half_na:
            print(f"- `{column}`")
            
        
        return info_df
    

    def visualize_missing_values(self):
        '''
        This method generates a heatmap to visually represent the missing values in the dataset.
        '''

        missing_cols = self.data.columns[self.data.isna().any()]

        missing_data = self.data[missing_cols]

        sns.heatmap(missing_data.T.isnull(), cbar=False)


    def visualize_outliers(self):
        '''
        This funcions helps in visualizing outliers using boxplot
        '''

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)

        for i in range(0, num_cols, 5):
            fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize =(20, 5))

            for j, col in enumerate(numerical_cols[i: i+5]):
                sns.boxplot(y=self.data[col], ax=axes[j])
                axes[j].set_title(col)

            plt.tight_layout()
            plt.show()



    def outlier_remover(self, columns: list) -> pd.DataFrame:
        '''
        This funtion removes all the outliers in a data using IQR technique

        Parameters: 
            columns(list): A list of columns that we don't need to remove the outlier, like unique identifiers

        Returns:
            pd.DataFrame: A dataframe without outliers
        '''

        numeric_col = self.data.select_dtypes(include='float64').columns
        fix_cols = [col for col in numeric_col if col not in columns]

        Q1 = self.data[fix_cols].quantile(0.25)
        Q3 = self.data[fix_cols].quantile(0.75)

        IQ = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQ
        upper_bound = Q3 + 1.5 * IQ

        self.data[fix_cols] = self.data[fix_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

        return self.data  
    


    def standardize_data(self, type: str):
        '''
        This function standardizes data:       
            * This function change all columns containing ms to sec
            * changes all columns from bytes to mega-bytes

        Parameters:
            * type(str): this specifies whether the columns to standardize are volumetric or time based
        '''
        
        if type == 'time':
            columns_to_standardize = list(self.data.columns[self.data.columns.str.contains('ms')])
            self.data[columns_to_standardize] = self.data[columns_to_standardize] / 1000

            new_cols = []
            for item in columns_to_standardize:
                if ')' in item:
                    new_cols.append(item.replace('ms', 's'))
                else:
                    new_cols.append(item.replace('ms', '(s)'))
        elif type == 'volume':
            columns_to_standardize = list(self.data.columns[self.data.columns.str.contains('Bytes')])
            self.data[columns_to_standardize] = self.data[columns_to_standardize] / (1024 * 1024)
            new_cols = []
            for item in columns_to_standardize:
                if ')' in item:
                    new_cols.append(item.replace('Bytes', 'MB'))
                else:
                    new_cols.append(item.replace('Bytes', '(MB)'))            


        self.data.rename(columns=dict(zip(columns_to_standardize, new_cols)), inplace=True)

        return self.data

      